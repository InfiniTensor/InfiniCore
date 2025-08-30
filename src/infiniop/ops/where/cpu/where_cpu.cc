#include "where_cpu.h"

namespace op::where::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &condition_desc = input_desc_vec.at(0);
    const auto &a_desc = input_desc_vec.at(1);
    const auto &b_desc = input_desc_vec.at(2);
    const auto &output_shape = out_desc->shape();
    const auto &condition_shape = condition_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    // Check condition tensor data type (should be bool as per competition.md)
    if (condition_desc->dtype() != INFINI_DTYPE_BOOL) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check that a and b have the same dtype as output
    if (a_desc->dtype() != dtype || b_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_DTYPE(dtype, 
                INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    // Check shapes are compatible (broadcast or same)
    CHECK_SAME_SHAPE(output_shape, a_shape);
    CHECK_SAME_SHAPE(output_shape, b_shape);
    CHECK_SAME_SHAPE(output_shape, condition_shape);

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

    // Execute where operation based on data type using heterogeneous input types
    // condition (bool), a (output_dtype), b (output_dtype) -> output (output_dtype)
    switch (_dtype) {
        case INFINI_DTYPE_F16:
            return _device_info->calculate<WhereOp, fp16_t, bool, fp16_t, fp16_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_F32:
            return _device_info->calculate<WhereOp, float, bool, float, float>(_info, output, inputs, stream);
        case INFINI_DTYPE_F64:
            return _device_info->calculate<WhereOp, double, bool, double, double>(_info, output, inputs, stream);
        case INFINI_DTYPE_BF16:
            return _device_info->calculate<WhereOp, bf16_t, bool, bf16_t, bf16_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_I8:
            return _device_info->calculate<WhereOp, int8_t, bool, int8_t, int8_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_I16:
            return _device_info->calculate<WhereOp, int16_t, bool, int16_t, int16_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_I32:
            return _device_info->calculate<WhereOp, int32_t, bool, int32_t, int32_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_I64:
            return _device_info->calculate<WhereOp, int64_t, bool, int64_t, int64_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_U8:
            return _device_info->calculate<WhereOp, uint8_t, bool, uint8_t, uint8_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_U16:
            return _device_info->calculate<WhereOp, uint16_t, bool, uint16_t, uint16_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_U32:
            return _device_info->calculate<WhereOp, uint32_t, bool, uint32_t, uint32_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_U64:
            return _device_info->calculate<WhereOp, uint64_t, bool, uint64_t, uint64_t>(_info, output, inputs, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::where::cpu