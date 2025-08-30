#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "cast_nvidia.cuh"

namespace op::cast::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto output_dtype = out_desc->dtype();
    auto input_dtype = input_desc_vec.at(0)->dtype();

    const auto &input_desc = input_desc_vec.at(0);
    const auto &out_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();

    CHECK_SAME_SHAPE(out_desc->shape(), input_desc_vec.at(0)->shape());
    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);
    auto info = info_result.take();
    auto workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void *);
    // Create DeviceImpl using the correct pattern from the macro
    auto device_impl_result = op::elementwise::nvidia::DeviceImpl::create(handle->internal());
    CHECK_RESULT(device_impl_result);

    // Create nvidia elementwise descriptor
    *desc_ptr = new Descriptor(
        input_dtype,
        output_dtype,
        std::move(info),
        std::move(device_impl_result.take()),
        workspace_size,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

#define SWITCH_IN_TYPE_NVIDIA(OUT_TYPE, IN_TYPE)                                                                              \
    switch (IN_TYPE) {                                                                                                        \
    case INFINI_DTYPE_I32:                                                                                                    \
        return _device_info->calculate<256, cuda::CastOp, OUT_TYPE, int32_t>(_info, workspace, output, inputs, stream);       \
    case INFINI_DTYPE_I64:                                                                                                    \
        return _device_info->calculate<256, cuda::CastOp, OUT_TYPE, int64_t>(_info, workspace, output, inputs, stream);       \
    case INFINI_DTYPE_U32:                                                                                                    \
        return _device_info->calculate<256, cuda::CastOp, OUT_TYPE, uint32_t>(_info, workspace, output, inputs, stream);      \
    case INFINI_DTYPE_U64:                                                                                                    \
        return _device_info->calculate<256, cuda::CastOp, OUT_TYPE, uint64_t>(_info, workspace, output, inputs, stream);      \
    case INFINI_DTYPE_F16:                                                                                                    \
        return _device_info->calculate<256, cuda::CastOp, OUT_TYPE, half>(_info, workspace, output, inputs, stream);          \
    case INFINI_DTYPE_F32:                                                                                                    \
        return _device_info->calculate<256, cuda::CastOp, OUT_TYPE, float>(_info, workspace, output, inputs, stream);         \
    case INFINI_DTYPE_F64:                                                                                                    \
        return _device_info->calculate<256, cuda::CastOp, OUT_TYPE, double>(_info, workspace, output, inputs, stream);        \
    case INFINI_DTYPE_BF16:                                                                                                   \
        return _device_info->calculate<256, cuda::CastOp, OUT_TYPE, cuda_bfloat16>(_info, workspace, output, inputs, stream); \
    default:                                                                                                                  \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                                                                                \
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
        SWITCH_IN_TYPE_NVIDIA(int32_t, _input_dtype)
    case INFINI_DTYPE_I64:
        SWITCH_IN_TYPE_NVIDIA(int64_t, _input_dtype)
    case INFINI_DTYPE_U32:
        SWITCH_IN_TYPE_NVIDIA(uint32_t, _input_dtype)
    case INFINI_DTYPE_U64:
        SWITCH_IN_TYPE_NVIDIA(uint64_t, _input_dtype)
    case INFINI_DTYPE_F16:
        SWITCH_IN_TYPE_NVIDIA(half, _input_dtype)
    case INFINI_DTYPE_F32:
        SWITCH_IN_TYPE_NVIDIA(float, _input_dtype)
    case INFINI_DTYPE_F64:
        SWITCH_IN_TYPE_NVIDIA(double, _input_dtype)
    case INFINI_DTYPE_BF16:
        SWITCH_IN_TYPE_NVIDIA(cuda_bfloat16, _input_dtype)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::cast::nvidia