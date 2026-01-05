#include "hardswish_cpu.h"

namespace op::hardswish::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // 校验输入是否存在
    const auto &input_desc = input_desc_vec.at(0);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();

    // HardSwish 支持的数据类型与 SiLU 一致
    CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    // 校验输入输出形状必须一致
    CHECK_SAME_SHAPE(output_shape, input_shape);

    // 使用 CPU Elementwise 通用宏创建描述符
    // 注意：这里直接复用 Elementwise 的逻辑
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // 根据数据类型分发计算
    // 关键点：将 SiluOp 替换为 HardSwishOp
    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<HardSwishOp, bf16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<HardSwishOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<HardSwishOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<HardSwishOp, double>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::hardswish::cpu