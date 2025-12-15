#include "where_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::where::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // Expected inputs: 0: cond, 1: x, 2: y
    if (input_desc_vec.size() != 3) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto out_dtype = out_desc->dtype();

    const auto &cond_desc = input_desc_vec.at(0);
    const auto &x_desc = input_desc_vec.at(1);
    const auto &y_desc = input_desc_vec.at(2);

    const auto &out_shape = out_desc->shape();
    const auto &cond_shape = cond_desc->shape();
    const auto &x_shape = x_desc->shape();
    const auto &y_shape = y_desc->shape();

    // cond must be bool
    CHECK_DTYPE(cond_desc->dtype(), INFINI_DTYPE_BOOL);

    // x, y and output must share the same dtype
    auto x_dtype = x_desc->dtype();
    auto y_dtype = y_desc->dtype();
    CHECK_OR_RETURN(x_dtype == y_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_dtype == out_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    // Supported value dtypes (extend if needed)
    CHECK_DTYPE(
        out_dtype,
        INFINI_DTYPE_F16,
        INFINI_DTYPE_F32,
        INFINI_DTYPE_F64,
        INFINI_DTYPE_BF16,
        INFINI_DTYPE_I32,
        INFINI_DTYPE_I64,
        INFINI_DTYPE_U8);

    // For now, require all shapes to match (no broadcasting)
    CHECK_SAME_SHAPE(out_shape, cond_shape, x_shape, y_shape);

    // Create CPU elementwise descriptor with output dtype
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, out_dtype, out_desc, input_desc_vec);

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
        return _device_info->calculate<WhereOp, fp16_t, bool, fp16_t, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<WhereOp, float, bool, float, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<WhereOp, double, bool, double, double>(_info, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<WhereOp, bf16_t, bool, bf16_t, bf16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<WhereOp, int32_t, bool, int32_t, int32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<WhereOp, int64_t, bool, int64_t, int64_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U8:
        return _device_info->calculate<WhereOp, uint8_t, bool, uint8_t, uint8_t>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::where::cpu


