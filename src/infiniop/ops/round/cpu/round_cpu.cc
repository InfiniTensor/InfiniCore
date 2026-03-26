#include "round_cpu.h"

#include <type_traits>

namespace op::round::cpu {

Descriptor::Descriptor(infiniDtype_t dtype,
                       op::elementwise::ElementwiseInfo info,
                       size_t workspace_size,
                       infiniDevice_t device_type,
                       int device_id,
                       int decimals)
    : InfiniopDescriptor{device_type, device_id},
      _dtype(dtype),
      _info(std::move(info)),
      _workspace_size(workspace_size),
      _decimals(decimals) {}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    int decimals) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &x_desc = input_desc_vec.at(0);
    const auto &y_shape = out_desc->shape();
    const auto &x_shape = x_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    CHECK_SAME_SHAPE(y_shape, x_shape);

    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(
        dtype, info_result.take(), 0,
        handle->device, handle->device_id, decimals);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
static infiniStatus_t launchCpuRound(const op::elementwise::ElementwiseInfo &info,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     int decimals) {
    if (inputs.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    T *out = reinterpret_cast<T *>(output);
    const T *in = reinterpret_cast<const T *>(inputs[0]);
    const auto ndim = info.getNdim();
    const auto *output_shape = info.getOutputShape();
    const auto *output_strides = info.getOutputStrides();
    const auto *input_shape = info.getInputShape(0);
    const auto *input_strides = info.getInputStrides(0);
    const auto *input_contiguous = info.getInputContiguous();
    ptrdiff_t output_size = info.getOutputSize();

#pragma omp parallel for if (output_size > 1024)
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        const size_t out_idx = info.isOutputContiguous()
                                 ? static_cast<size_t>(i)
                                 : op::common_cpu::indexToOffset(i, ndim, output_shape, output_strides);
        const size_t in_idx = input_contiguous[0]
                                ? static_cast<size_t>(i)
                                : op::common_cpu::indexToOffset(i, ndim, input_shape, input_strides);

        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float value = utils::cast<float>(in[in_idx]);
            float rounded = RoundOp{}(value, decimals);
            out[out_idx] = utils::cast<T>(rounded);
        } else {
            out[out_idx] = RoundOp{}(in[in_idx], decimals);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    (void)stream;

    if (inputs.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return launchCpuRound<fp16_t>(_info, output, inputs, _decimals);
    case INFINI_DTYPE_F32:
        return launchCpuRound<float>(_info, output, inputs, _decimals);
    case INFINI_DTYPE_F64:
        return launchCpuRound<double>(_info, output, inputs, _decimals);
    case INFINI_DTYPE_BF16:
        return launchCpuRound<bf16_t>(_info, output, inputs, _decimals);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::round::cpu