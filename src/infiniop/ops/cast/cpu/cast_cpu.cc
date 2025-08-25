#include "cast_cpu.h"

namespace op::cast::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto input_dtype = input_desc_vec.at(0)->dtype();
    auto output_dtype = out_desc->dtype();

    CHECK_SAME_SHAPE(out_desc->shape(), input_desc_vec.at(0)->shape());
    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);
    // create CPU elementwise descriptor
    *desc_ptr = new Descriptor(
        input_dtype,
        output_dtype,
        info_result.take(),
        nullptr,
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

#define SWITCH_IN_TYPE(OUT_TYPE, IN_TYPE)                                                          \
    switch (IN_TYPE) {                                                                             \
    case INFINI_DTYPE_I32:                                                                         \
        return _device_info->calculate<CastOp, OUT_TYPE, int32_t>(_info, output, inputs, stream);  \
    case INFINI_DTYPE_I64:                                                                         \
        return _device_info->calculate<CastOp, OUT_TYPE, int64_t>(_info, output, inputs, stream);  \
    case INFINI_DTYPE_U32:                                                                         \
        return _device_info->calculate<CastOp, OUT_TYPE, uint32_t>(_info, output, inputs, stream); \
    case INFINI_DTYPE_U64:                                                                         \
        return _device_info->calculate<CastOp, OUT_TYPE, uint64_t>(_info, output, inputs, stream); \
    case INFINI_DTYPE_F16:                                                                         \
        return _device_info->calculate<CastOp, OUT_TYPE, fp16_t>(_info, output, inputs, stream);   \
    case INFINI_DTYPE_F32:                                                                         \
        return _device_info->calculate<CastOp, OUT_TYPE, float>(_info, output, inputs, stream);    \
    case INFINI_DTYPE_F64:                                                                         \
        return _device_info->calculate<CastOp, OUT_TYPE, double>(_info, output, inputs, stream);   \
    case INFINI_DTYPE_BF16:                                                                        \
        return _device_info->calculate<CastOp, OUT_TYPE, bf16_t>(_info, output, inputs, stream);   \
    default:                                                                                       \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                                                     \
    }

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // Handle type conversions based on input and output types
    switch (_output_dtype) {
    case INFINI_DTYPE_I32:
        SWITCH_IN_TYPE(int32_t, _input_dtype)
    case INFINI_DTYPE_I64:
        SWITCH_IN_TYPE(int64_t, _input_dtype)
    case INFINI_DTYPE_U32:
        SWITCH_IN_TYPE(uint32_t, _input_dtype)
    case INFINI_DTYPE_U64:
        SWITCH_IN_TYPE(uint64_t, _input_dtype)
    case INFINI_DTYPE_F16:
        SWITCH_IN_TYPE(fp16_t, _input_dtype)
    case INFINI_DTYPE_F32:
        SWITCH_IN_TYPE(float, _input_dtype)
    case INFINI_DTYPE_F64:
        SWITCH_IN_TYPE(double, _input_dtype)
    case INFINI_DTYPE_BF16:
        SWITCH_IN_TYPE(bf16_t, _input_dtype)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::cast::cpu