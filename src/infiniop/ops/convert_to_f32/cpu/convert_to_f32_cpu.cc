#include "convert_to_f32_cpu.h"

namespace op::convert_to_f32::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto out_dtype = out_desc->dtype();
    const auto &x_desc = input_desc_vec.at(0);
    auto in_dtype = x_desc->dtype();

    CHECK_DTYPE(out_dtype, INFINI_DTYPE_F32);
    CHECK_DTYPE(in_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

    const auto &y_shape = out_desc->shape();
    const auto &x_shape = x_desc->shape();
    CHECK_SAME_SHAPE(y_shape, x_shape);

    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, in_dtype, out_desc, input_desc_vec);

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
        return _device_info->calculate<ConvertToF32Op, float, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<ConvertToF32Op, float, bf16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<ConvertToF32Op, float, float>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::convert_to_f32::cpu
