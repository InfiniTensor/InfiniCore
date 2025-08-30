#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>
#include "../../../reduce/cuda/reduce.cuh"
#include "batch_norm_backward_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::batch_norm_backward::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * input,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * running_mean,
    const Tdata * running_var,
    size_t batch_size,
    size_t channel_size,
    size_t dim_size,
    ptrdiff_t grad_weight_stride,
    ptrdiff_t grad_bias_stride,
    ptrdiff_t weight_stride,
    ptrdiff_t running_mean_stride,
    ptrdiff_t running_var_stride    
) {
    batchNormBackwardKernel<BLOCK_SIZE, Tdata, Tcompute>(
        grad_input,
        grad_weight,
        grad_bias,
        input,
        grad_output,
        weight,
        running_mean,
        running_var,
        batch_size,
        channel_size,
        dim_size,
        grad_weight_stride,
        grad_bias_stride,
        weight_stride,
        running_mean_stride,
        running_var_stride            
    );
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_batch_norm_backward(
    const BatchNormBackwardInfo &info,
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * input,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * running_mean,
    const Tdata * running_var,
    cudaStream_t stream
) {
    launchKernel<1, Tdata, double><<<info.channel_size, 1, 0, stream>>>(
        grad_input,
        grad_weight,
        grad_bias,
        input,
        grad_output,
        weight,
        running_mean,
        running_var,
        info.batch_size,
        info.channel_size,
        info.dim_size,
        info.grad_weight_stride,
        info.grad_bias_stride,
        info.weight_stride,
        info.running_mean_stride,
        info.running_var_stride
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
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_weight_desc,
    infiniopTensorDescriptor_t grad_bias_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = grad_input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = BatchNormBackwardInfo::createBatchNormBackwardInfo(
        grad_input_desc,
        grad_weight_desc,
        grad_bias_desc,
        input_desc,
        grad_output_desc,
        weight_desc,
        running_mean_desc,
        running_var_desc
    );
    CHECK_RESULT(result);
    const BatchNormBackwardInfo &info = result.take();
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
    void * grad_input,
    void * grad_weight,
    void * grad_bias,
    const void * input,
    const void * grad_output,
    const void * weight,
    const void * running_mean,
    const void * running_var,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    #define CALCULATE_BATCH_NORM_BACKWARD(BLOCK_SIZE, TDATA) \
        calculate_batch_norm_backward<BLOCK_SIZE, TDATA>(_info, (TDATA *)grad_input, (TDATA *)grad_weight, (TDATA *)grad_bias, (const TDATA *)input, (const TDATA *)grad_output, (const TDATA *)weight, (const TDATA *)running_mean, (const TDATA *)running_var, stream)
    #define CALCULATE_BATCH_NORM_BACKWARD_WITH_BLOCK_SIZE(BLOCK_SIZE)           \
    {                                                                           \
        if (_info.dtype == INFINI_DTYPE_F16)                                    \
            return CALCULATE_BATCH_NORM_BACKWARD(BLOCK_SIZE, half);             \
        else if (_info.dtype == INFINI_DTYPE_F32)                               \
            return CALCULATE_BATCH_NORM_BACKWARD(BLOCK_SIZE, float);            \
        else if (_info.dtype == INFINI_DTYPE_BF16)                              \
            return CALCULATE_BATCH_NORM_BACKWARD(BLOCK_SIZE, __nv_bfloat16);    \
        else                                                                    \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                              \
    }
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_BATCH_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_BATCH_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096)
        CALCULATE_BATCH_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    return INFINI_STATUS_SUCCESS;
}
} // namespace op:batch_norm_backward::nvidia
