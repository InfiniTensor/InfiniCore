#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cub/block/block_reduce.cuh>
#include "../../../reduce/cuda/reduce.cuh"
#include "../cuda/kernel.cuh"
#include "rms_norm_backward_nvidia.h"
#include "../info.h"

namespace op::rms_norm_backward::nvidia {

using namespace op::rms_norm_backward::cuda;

/**
 * @brief Launch wrapper for RMS Norm backward kernel
 * 
 * This function provides a type-safe interface to launch the CUDA kernel
 * for RMS normalization backward pass computation.
 */
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute, typename Tweight>
static infiniStatus_t launchRmsNormBackwardKernel(
    const RMSNormBackwardInfo &info,
    Tdata * grad_x,
    Tcompute * grad_w_cuda,
    const Tdata * grad_y,
    const Tdata * x,
    const Tweight * w,
    cudaStream_t stream,
    void * workspace
) {
    size_t ndim = info.ndim();
    
    // Prepare stride and shape arrays in device memory
    size_t * shape_cuda = reinterpret_cast<size_t*>(workspace);
    ptrdiff_t * grad_x_strides_cuda = reinterpret_cast<ptrdiff_t*>(shape_cuda + ndim);
    ptrdiff_t * grad_y_strides_cuda = grad_x_strides_cuda + ndim;
    ptrdiff_t * x_strides_cuda = grad_y_strides_cuda + ndim;

    // Copy stride and shape data to device
    CHECK_CUDA(cudaMemcpyAsync(shape_cuda, info.shape.data(), 
                               sizeof(size_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(grad_x_strides_cuda, info.grad_x_strides.data(), 
                               sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(grad_y_strides_cuda, info.grad_y_strides.data(), 
                               sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(x_strides_cuda, info.x_strides.data(), 
                               sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));

    // Launch the main backward kernel
    rmsNormBackwardKernel<BLOCK_SIZE, Tdata, Tcompute, Tweight><<<info.batch_size(), BLOCK_SIZE, 0, stream>>>(
        grad_x, grad_w_cuda, grad_y, x, w,
        ndim, info.batch_size(), info.dim(),
        grad_x_strides_cuda, shape_cuda,
        grad_y_strides_cuda, x_strides_cuda,
        info.w_strides[0],
        info.epsilon
    );
    
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

/**
 * @brief Launch wrapper for weight gradient summation kernel
 */
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
static infiniStatus_t launchSumUpGradWKernel(
    const RMSNormBackwardInfo &info,
    Tdata * grad_w,
    Tcompute * grad_w_cuda,
    cudaStream_t stream
) {
    sumUpGradWKernel<BLOCK_SIZE, Tdata, Tcompute><<<info.dim(), BLOCK_SIZE, 0, stream>>>(
        grad_w, grad_w_cuda, info.batch_size(), info.grad_w_strides[0]
    );
    
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

/**
 * @brief Main computation function for RMS Norm backward pass
 * 
 * This function orchestrates the two-stage computation:
 * 1. Compute input gradients and intermediate weight gradients
 * 2. Sum up weight gradients across batches
 */
template<unsigned int BLOCK_SIZE, typename Tdata, typename Tweight>
static infiniStatus_t calculate_rms_norm_backward(
    const RMSNormBackwardInfo &info,
    Tdata * grad_x,
    Tdata * grad_w,
    const Tdata * grad_y,
    const Tdata * x,
    const Tweight * w,
    cudaStream_t stream,
    void * workspace
) {
    // Calculate workspace layout
    size_t ndim = info.ndim();
    size_t stride_workspace_size = sizeof(ptrdiff_t) * ndim * 4;
    float * grad_w_cuda = reinterpret_cast<float *>(
        reinterpret_cast<char*>(workspace) + stride_workspace_size
    );

    // Stage 1: Compute input gradients and intermediate weight gradients
    auto status1 = launchRmsNormBackwardKernel<BLOCK_SIZE, Tdata, float, Tweight>(
        info, grad_x, grad_w_cuda, grad_y, x, w, stream, workspace
    );
    CHECK_STATUS(status1);

    // Stage 2: Sum up weight gradients across batches
    auto status2 = launchSumUpGradWKernel<BLOCK_SIZE, Tdata, float>(
        info, grad_w, grad_w_cuda, stream
    );
    CHECK_STATUS(status2);
    
    return INFINI_STATUS_SUCCESS;
}
//  ------------------------------------ end: call launchKernel ------------------------------------

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_x_desc,
    infiniopTensorDescriptor_t grad_w_desc,
    infiniopTensorDescriptor_t grad_y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto info_result = RMSNormBackwardInfo::createRMSNormBackwardInfo(grad_x_desc, grad_w_desc, grad_y_desc, x_desc, w_desc, epsilon);
    if (!info_result) {
        return info_result.status();
    }
    RMSNormBackwardInfo info = info_result.take();

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    size_t workspace_size = sizeof(ptrdiff_t) * info.ndim() * 4 + sizeof(float) * info.batch_size() * info.dim();

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        std::move(info),
        workspace_size,
        handle_->device, handle_->device_id);
    return INFINI_STATUS_SUCCESS;
}



infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *grad_x,
    void *grad_w,
    const void *grad_y,
    const void *x,
    const void *w,
    void *stream) const {
    
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // Use the two-stage architecture with proper data type handling
    if (_info.grad_x_dtype == INFINI_DTYPE_F16 && _info.w_dtype == INFINI_DTYPE_F16) {
        return calculate_rms_norm_backward<1024, half, half>(
            _info, 
            reinterpret_cast<half*>(grad_x),
            reinterpret_cast<half*>(grad_w),
            reinterpret_cast<const half*>(grad_y),
            reinterpret_cast<const half*>(x),
            reinterpret_cast<const half*>(w),
            cuda_stream,
            workspace
        );
    } else if (_info.grad_x_dtype == INFINI_DTYPE_F16 && _info.w_dtype == INFINI_DTYPE_F32) {
        return calculate_rms_norm_backward<1024, half, float>(
            _info,
            reinterpret_cast<half*>(grad_x),
            reinterpret_cast<half*>(grad_w),
            reinterpret_cast<const half*>(grad_y),
            reinterpret_cast<const half*>(x),
            reinterpret_cast<const float*>(w),
            cuda_stream,
            workspace
        );
    } else if (_info.grad_x_dtype == INFINI_DTYPE_BF16 && _info.w_dtype == INFINI_DTYPE_BF16) {
        return calculate_rms_norm_backward<1024, __nv_bfloat16, __nv_bfloat16>(
            _info,
            reinterpret_cast<__nv_bfloat16*>(grad_x),
            reinterpret_cast<__nv_bfloat16*>(grad_w),
            reinterpret_cast<const __nv_bfloat16*>(grad_y),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(w),
            cuda_stream,
            workspace
        );
    } else if (_info.grad_x_dtype == INFINI_DTYPE_BF16 && _info.w_dtype == INFINI_DTYPE_F32) {
        return calculate_rms_norm_backward<1024, __nv_bfloat16, float>(
            _info,
            reinterpret_cast<__nv_bfloat16*>(grad_x),
            reinterpret_cast<__nv_bfloat16*>(grad_w),
            reinterpret_cast<const __nv_bfloat16*>(grad_y),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const float*>(w),
            cuda_stream,
            workspace
        );
    } else if (_info.grad_x_dtype == INFINI_DTYPE_F32 && _info.w_dtype == INFINI_DTYPE_F32) {
        return calculate_rms_norm_backward<1024, float, float>(
            _info,
            reinterpret_cast<float*>(grad_x),
            reinterpret_cast<float*>(grad_w),
            reinterpret_cast<const float*>(grad_y),
            reinterpret_cast<const float*>(x),
            reinterpret_cast<const float*>(w),
            cuda_stream,
            workspace
        );
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::rms_norm_backward::nvidia

// Template instantiation is handled by the DESCRIPTOR macro