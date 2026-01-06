#include "hardtanh_cpu.h"

namespace op::hardtanh::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float min_val,   // 新增参数
    float max_val) { // 新增参数

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &input_desc = input_desc_vec.at(0);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    CHECK_SAME_SHAPE(output_shape, input_shape);

    // 创建 CPU elementwise descriptor
    // 注意：这里需要确保你的宏或基类 Descriptor 能存下 min/max 值
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    // 将参数保存到 descriptor 实例中
    auto desc = *desc_ptr;
    desc->min_val = min_val;
    desc->max_val = max_val;

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // 实例化带有参数的 Op
    HardTanhOp op(this->min_val, this->max_val);

    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<HardTanhOp, bf16_t>(_info, output, inputs, stream, op);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<HardTanhOp, fp16_t>(_info, output, inputs, stream, op);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<HardTanhOp, float>(_info, output, inputs, stream, op);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<HardTanhOp, double>(_info, output, inputs, stream, op);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::hardtanh::cpu