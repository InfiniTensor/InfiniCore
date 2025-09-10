#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "leaky_relu_nvidia.cuh"

namespace op::leaky_relu::nvidia {

Descriptor::~Descriptor() = default;
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float negative_slope) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &x_desc = input_desc_vec.at(0);
    const auto &y_shape = out_desc->shape();
    const auto &x_shape = x_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    CHECK_SAME_SHAPE(y_shape, x_shape);

    // create NVIDIA elementwise descriptor
    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);
    auto info = info_result.take();
    auto workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void *) + sizeof(float); // device negative_slope
    auto device_impl_result = op::elementwise::nvidia::DeviceImpl::create(handle->internal());
    CHECK_RESULT(device_impl_result);

    *desc_ptr = new Descriptor(
        dtype,
        std::move(info),
        std::move(device_impl_result.take()),
        workspace_size,
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

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    const int8_t *d_negative_slope_start = reinterpret_cast<int8_t *>(workspace) + workspace_size - sizeof(_negative_slope);
    CHECK_CUDA(cudaMemcpyAsync((void *)d_negative_slope_start,
                               &_negative_slope,
                               sizeof(_negative_slope),
                               cudaMemcpyHostToDevice,
                               reinterpret_cast<cudaStream_t>(stream)));
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::LeakyReLUOp, half>(_info, workspace, output, inputs, stream, reinterpret_cast<const float *>(d_negative_slope_start));
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::LeakyReLUOp, cuda_bfloat16>(_info, workspace, output, inputs, stream, reinterpret_cast<const float *>(d_negative_slope_start));
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::LeakyReLUOp, float>(_info, workspace, output, inputs, stream, reinterpret_cast<const float *>(d_negative_slope_start));
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::LeakyReLUOp, double>(_info, workspace, output, inputs, stream, reinterpret_cast<const float *>(d_negative_slope_start));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::leaky_relu::nvidia