#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cub/block/block_reduce.cuh>
#include "../../../reduce/cuda/reduce.cuh"
#include "../cuda/kernel.cuh"
#include "rms_norm_backward_nvidia.cuh"
#include "../info.h"

namespace op::rms_norm_backward::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata * grad_x,
    Tdata * grad_w_cuda,
    const Tdata * grad_y,
    const Tdata * x,
    const Tdata * w,
    size_t ndim,
    size_t batch_size,
    size_t norm_size,
    const ptrdiff_t * grad_x_strides,
    const ptrdiff_t * grad_w_strides,
    const ptrdiff_t * grad_y_strides,
    const ptrdiff_t * x_strides,
    ptrdiff_t  w_stride
) {
    rmsNormBackwardKernel<BLOCK_SIZE, Tdata, Tcompute>(
        grad_x,
        grad_w_cuda,
        grad_y,
        x,
        w,
        ndim,
        batch_size,
        norm_size,
        grad_x_strides,
        grad_w_strides,
        grad_y_strides,
        x_strides,
        w_stride
    );
}
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL sumUpGradWLaunchKernel(
    Tdata * grad_w,
    Tdata * grad_w_cuda,
    size_t batch_size,
    ptrdiff_t grad_w_stride    
) {
    sumUpGradWKernel<BLOCK_SIZE, Tdata, Tcompute>(grad_w, grad_w_cuda, batch_size, grad_w_stride);
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_rms_norm_backward(
    const RMSNormBackwardInfo &info,
    Tdata * grad_x,
    Tdata * grad_w,
    const Tdata * grad_y,
    const Tdata * x,
    const Tdata * w,
    cudaStream_t stream,
    void * workspace
) {
    size_t ndim = info.ndim;
    ptrdiff_t * contiguous_strides = new ptrdiff_t[ndim - 1];
    size_t last_dim = 1, last_stride = 1;
    for (size_t d = 0; d < ndim - 1; d ++)
    {
        contiguous_strides[d] = last_dim * last_stride;  
        last_dim = info.x_shape[d];
        last_stride = contiguous_strides[d];
    }


    ptrdiff_t * contiguous_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace);
    ptrdiff_t * grad_x_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace) + ndim;
    ptrdiff_t * grad_y_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace) + ndim * 2;
    ptrdiff_t * x_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace) + ndim * 3;
    Tdata * grad_w_cuda = reinterpret_cast<Tdata *>(x_strides_cuda + ndim);

    CHECK_CUDA(cudaMemcpyAsync(contiguous_strides_cuda, contiguous_strides, sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(grad_x_strides_cuda, info.grad_x_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(grad_y_strides_cuda, info.grad_y_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(x_strides_cuda, info.x_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));    

    launchKernel<1, Tdata, float><<<info.batch_size, 1, 0, stream>>>(
        grad_x,
        grad_w_cuda,
        grad_y,
        x,
        w,
        info.ndim,
        info.batch_size,
        info.norm_size,
        grad_x_strides_cuda,
        grad_y_strides_cuda,
        x_strides_cuda,
        contiguous_strides_cuda,
        info.w_strides[0]
    );
    cudaDeviceSynchronize();
    sumUpGradWLaunchKernel<BLOCK_SIZE, Tdata, float><<<info.norm_size, BLOCK_SIZE, 0, stream>>>(
        grad_w,
        grad_w_cuda,
        info.batch_size,
        info.grad_w_strides[0]
    );
    delete[] contiguous_strides;
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
    infiniopTensorDescriptor_t w_desc
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = grad_x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = RMSNormBackwardInfo::createRMSNormBackwardInfo(
        grad_x_desc,
        grad_w_desc,
        grad_y_desc,
        x_desc,
        w_desc
    );
    CHECK_RESULT(result);
    const RMSNormBackwardInfo &info = result.take();

    size_t WorkSpaceSize = sizeof(ptrdiff_t) * info.ndim * 4 + info.total_size * infiniSizeOf(dtype);

    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        new Opaque{handle->internal()},
        handle->device, handle->device_id
    );    
    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void * workspace,
    size_t workspace_size,
    void * grad_x,
    void * grad_w,
    const void * grad_y,
    const void * x,
    const void * w,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    #define CALCULATE_RMS_NORM_BACKWARD(BLOCK_SIZE, TDATA) \
        calculate_rms_norm_backward<BLOCK_SIZE, TDATA>(_info, (TDATA *)grad_x, (TDATA *)grad_w, (const TDATA *)grad_y, (const TDATA *)x, (const TDATA *)w, stream, workspace)
    #define CALCULATE_RMS_NORM_BACKWARD_WITH_BLOCK_SIZE(BLOCK_SIZE)             \
    {                                                                           \
        if (_info.dtype == INFINI_DTYPE_F16)                                    \
            return CALCULATE_RMS_NORM_BACKWARD(BLOCK_SIZE, half);               \
        else if (_info.dtype == INFINI_DTYPE_F32)                               \
            return CALCULATE_RMS_NORM_BACKWARD(BLOCK_SIZE, float);              \
        else if (_info.dtype == INFINI_DTYPE_BF16)                              \
            return CALCULATE_RMS_NORM_BACKWARD(BLOCK_SIZE, __nv_bfloat16);      \
        else                                                                    \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                              \
    }
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_RMS_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_RMS_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096)
        CALCULATE_RMS_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rms_norm_backward::nvidia
