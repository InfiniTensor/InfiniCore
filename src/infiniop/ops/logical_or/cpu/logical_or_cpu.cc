#include "logical_or_cpu.h"
#include "infinicore.h"

namespace op::logical_or::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    auto dtype = a_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_BOOL:
        return _device_info->calculate<LogicalOrOp, bool, bool, bool>(_info, output, inputs, stream);
    case INFINI_DTYPE_I8:
        return _device_info->calculate<LogicalOrOp, bool, int8_t, int8_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I16:
        return _device_info->calculate<LogicalOrOp, bool, int16_t, int16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<LogicalOrOp, bool, int32_t, int32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<LogicalOrOp, bool, int64_t, int64_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<LogicalOrOp, bool, bf16_t, bf16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<LogicalOrOp, bool, fp16_t, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<LogicalOrOp, bool, float, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<LogicalOrOp, bool, double, double>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::logical_or::cpu
