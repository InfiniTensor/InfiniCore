#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "or_nvidia.cuh"

namespace op::or_op::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    // Or operator supports bool and numeric types (following torch.logical_or)
    CHECK_DTYPE(dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // create CUDA elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_BOOL:
        return _device_info->calculate<256, cuda::OrOp, bool>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I8:
        return _device_info->calculate<256, cuda::OrOp, int8_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I16:
        return _device_info->calculate<256, cuda::OrOp, int16_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<256, cuda::OrOp, int32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<256, cuda::OrOp, int64_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U8:
        return _device_info->calculate<256, cuda::OrOp, uint8_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U16:
        return _device_info->calculate<256, cuda::OrOp, uint16_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U32:
        return _device_info->calculate<256, cuda::OrOp, uint32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U64:
        return _device_info->calculate<256, cuda::OrOp, uint64_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::OrOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::OrOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::OrOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::OrOp, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::or_op::nvidia