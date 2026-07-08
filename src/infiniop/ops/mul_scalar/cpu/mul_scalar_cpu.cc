#include "mul_scalar_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::mul_scalar::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = MulScalarInfo::create(output_desc, input_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        nullptr,
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
inline T mulScalarHost(T value, double alpha) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<T>(utils::cast<float>(value) * static_cast<float>(alpha));
    } else {
        return value * static_cast<T>(alpha);
    }
}

template <typename T>
infiniStatus_t calculateMulScalar(const MulScalarInfo &info, T *output, const T *input, double alpha) {
    const auto &elementwise_info = info.elementwise_info;
    const ptrdiff_t output_size = elementwise_info.getOutputSize();

    if (info.contiguous) {
#pragma omp parallel for if (output_size > 1024)
        for (ptrdiff_t i = 0; i < output_size; ++i) {
            output[i] = mulScalarHost(input[i], alpha);
        }
        return INFINI_STATUS_SUCCESS;
    }

#pragma omp parallel for if (output_size > 1024)
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        size_t out_idx = elementwise_info.isOutputContiguous()
                           ? i
                           : op::common_cpu::indexToOffset(
                               i,
                               elementwise_info.getNdim(),
                               elementwise_info.getOutputShape(),
                               elementwise_info.getOutputStrides());
        size_t in_idx = elementwise_info.getInputContiguous()[0]
                          ? i
                          : op::common_cpu::indexToOffset(
                              i,
                              elementwise_info.getNdim(),
                              elementwise_info.getInputShape(0),
                              elementwise_info.getInputStrides(0));
        output[out_idx] = mulScalarHost(input[in_idx], alpha);
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    double alpha,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return calculateMulScalar(_info, static_cast<fp16_t *>(output), static_cast<const fp16_t *>(input), alpha);
    case INFINI_DTYPE_BF16:
        return calculateMulScalar(_info, static_cast<bf16_t *>(output), static_cast<const bf16_t *>(input), alpha);
    case INFINI_DTYPE_F32:
        return calculateMulScalar(_info, static_cast<float *>(output), static_cast<const float *>(input), alpha);
    case INFINI_DTYPE_F64:
        return calculateMulScalar(_info, static_cast<double *>(output), static_cast<const double *>(input), alpha);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::mul_scalar::cpu
