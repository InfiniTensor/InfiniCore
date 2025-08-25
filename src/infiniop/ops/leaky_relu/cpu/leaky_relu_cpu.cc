#include "leaky_relu_cpu.h"

namespace op::leaky_relu::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float negative_slope) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &x_desc = input_desc_vec.at(0);
    const auto &y_shape = out_desc->shape();
    const auto &x_shape = x_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    CHECK_SAME_SHAPE(y_shape, x_shape);

    // create CPU elementwise descriptor
    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(
        dtype,
        info_result.take(),
        nullptr,
        0,
        handle->device,
        handle->device_id,
        negative_slope);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<LeakyReLUOp, fp16_t>(_info, output, inputs, stream, _negative_slope);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<LeakyReLUOp, float>(_info, output, inputs, stream, _negative_slope);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<LeakyReLUOp, double>(_info, output, inputs, stream, _negative_slope);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<LeakyReLUOp, bf16_t>(_info, output, inputs, stream, _negative_slope);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::leaky_relu::cpu