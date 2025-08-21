#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "linear_backward_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::linear_backward::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata * grad_x,
    Tdata * grad_w,
    Tdata * grad_b,
    const Tdata * grad_y,
    const Tdata * x,
    const Tdata * w,
    size_t out_features,
    ptrdiff_t grad_x_stride,
    ptrdiff_t grad_w_stride_out,
    ptrdiff_t grad_w_stride_in,
    ptrdiff_t grad_b_stride,
    ptrdiff_t grad_y_stride,
    ptrdiff_t x_stride,
    ptrdiff_t w_stride_out,
    ptrdiff_t w_stride_in,
    bool bias 
) {
    linearBackwardKernel<BLOCK_SIZE, Tdata, Tcompute>(
        grad_x,
        grad_w,
        grad_b,
        grad_y,
        x,
        w,
        out_features,
        grad_x_stride,
        grad_w_stride_out,
        grad_w_stride_in,
        grad_b_stride,
        grad_y_stride,
        x_stride,
        w_stride_out,
        w_stride_in,
        bias
    );
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_linear_backward(
    const LinearBackwardInfo &info,
    Tdata * grad_x,
    Tdata * grad_w,
    Tdata * grad_b,
    const Tdata * grad_y,
    const Tdata * x,
    const Tdata * w,
    cudaStream_t stream
) {
    launchKernel<1, Tdata, float><<<info.in_features, 1, 0, stream>>>(
        grad_x,
        grad_w,
        grad_b,
        grad_y,
        x,
        w,
        info.out_features,
        info.grad_x_stride,
        info.grad_w_stride_out,
        info.grad_w_stride_in,
        info.grad_b_stride,
        info.grad_y_stride,
        info.x_stride,
        info.w_stride_out,
        info.w_stride_in,
        info.bias
    );
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
    infiniopTensorDescriptor_t grad_b_desc,
    infiniopTensorDescriptor_t grad_y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = grad_x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = LinearBackwardInfo::createLinearBackwardInfo(
        grad_x_desc,
        grad_w_desc,
        grad_b_desc,
        grad_y_desc,
        x_desc,
        w_desc
    );
    CHECK_RESULT(result);
    const LinearBackwardInfo &info = result.take();
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
    void * grad_b,
    const void * grad_y,
    const void * x,
    const void * w,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    #define CALCULATE_LINEAR_BACKWARD(BLOCK_SIZE, TDATA) \
        calculate_linear_backward<BLOCK_SIZE, TDATA>(_info, (TDATA *)grad_x, (TDATA *)grad_w, (TDATA *)grad_b, (const TDATA *)grad_y, (const TDATA *)x, (const TDATA *)w, stream)
    #define CALCULATE_LINEAR_BACKWARD_WITH_BLOCK_SIZE(BLOCK_SIZE)               \
    {                                                                           \
        if (_info.dtype == INFINI_DTYPE_F16)                                    \
            return CALCULATE_LINEAR_BACKWARD(BLOCK_SIZE, half);                 \
        else if (_info.dtype == INFINI_DTYPE_F32)                               \
            return CALCULATE_LINEAR_BACKWARD(BLOCK_SIZE, float);                \
        else if (_info.dtype == INFINI_DTYPE_BF16)                              \
            return CALCULATE_LINEAR_BACKWARD(BLOCK_SIZE, __nv_bfloat16);        \
        else                                                                    \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                              \
    }
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_LINEAR_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_LINEAR_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096)
        CALCULATE_LINEAR_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    return INFINI_STATUS_SUCCESS;

    #undef CALCULATE_LINEAR_BACKWARD_WITH_BLOCK_SIZE
    #undef CALCULATE_LINEAR_BACKWARD    
}
} // namespace op::linear_backward::nvidia
