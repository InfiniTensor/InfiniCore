#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "layer_norm_backward_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::layer_norm_backward::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchStepOneKernel(
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * input_standardization,
    const Tdata * input_std_deviation,   
    size_t batch_size,
    size_t channel_size,
    size_t feature_size,
    ptrdiff_t grad_output_stride_b,
    ptrdiff_t grad_output_stride_c,
    ptrdiff_t input_standardization_stride_b,
    ptrdiff_t input_standardization_stride_c,
    ptrdiff_t input_std_deviation_stride_b,
    ptrdiff_t input_std_deviation_stride_c,
    ptrdiff_t grad_weight_stride,
    ptrdiff_t grad_bias_stride,
    bool bias
) {
    layerNormBackwardStepOneKernel<BLOCK_SIZE, Tdata, Tcompute>(
        grad_input,
        grad_weight,
        grad_bias,
        grad_output,
        weight,
        input_standardization,
        input_std_deviation,
        batch_size,
        channel_size,
        feature_size,
        grad_output_stride_b,
        grad_output_stride_c,
        input_standardization_stride_b,
        input_standardization_stride_c,
        input_std_deviation_stride_b,
        input_std_deviation_stride_c,
        grad_weight_stride,
        grad_bias_stride,
        bias        
    );
}

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchStepTwoKernel(
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * input_standardization,
    const Tdata * input_std_deviation, 
    size_t batch_size,
    size_t channel_size,
    size_t feature_size,
    ptrdiff_t grad_input_stride_b,
    ptrdiff_t grad_input_stride_c,
    ptrdiff_t grad_output_stride_b,
    ptrdiff_t grad_output_stride_c,  
    ptrdiff_t weight_stride,
    ptrdiff_t input_standardization_stride_b,
    ptrdiff_t input_standardization_stride_c,
    ptrdiff_t input_std_deviation_stride_b,
    ptrdiff_t input_std_deviation_stride_c  
) {
    layerNormBackwardStepTwoKernel<BLOCK_SIZE, Tdata, Tcompute>(
        grad_input,
        grad_weight,
        grad_bias,
        grad_output,
        weight,
        input_standardization,
        input_std_deviation,
        batch_size,
        channel_size,
        feature_size,
        grad_input_stride_b,
        grad_input_stride_c,
        grad_output_stride_b,
        grad_output_stride_c,  
        weight_stride,
        input_standardization_stride_b,
        input_standardization_stride_c,
        input_std_deviation_stride_b,
        input_std_deviation_stride_c
    );
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_layer_norm_backward(
    const LayerNormBackwardInfo &info,
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * input_standardization,
    const Tdata * input_std_deviation,
    cudaStream_t stream
) {
    launchStepOneKernel<1, Tdata, float><<<info.feature_size, 1, 0, stream>>>(
        grad_input,
        grad_weight,
        grad_bias,
        grad_output,
        weight,
        input_standardization,
        input_std_deviation,
        info.batch_size,
        info.channel_size,
        info.feature_size,
        info.grad_output_strides[0],
        info.grad_output_strides[1],
        info.input_standardization_strides[0],
        info.input_standardization_strides[1],
        info.input_std_deviation_strides[0],
        info.input_std_deviation_strides[1],
        info.grad_weight_strides[0],
        info.bias ? info.grad_bias_strides[0] : 0,
        info.bias
    );
    cudaDeviceSynchronize();
    launchStepTwoKernel<BLOCK_SIZE, Tdata, float><<<dim3(info.batch_size, info.channel_size), BLOCK_SIZE, 0, stream>>>(
        grad_input,
        grad_weight,
        grad_bias,
        grad_output,
        weight,
        input_standardization,
        input_std_deviation,
        info.batch_size,
        info.channel_size,
        info.feature_size,
        info.grad_input_strides[0],
        info.grad_input_strides[1],        
        info.grad_output_strides[0],
        info.grad_output_strides[1],
        info.weight_strides[0],
        info.input_standardization_strides[0],
        info.input_standardization_strides[1],
        info.input_std_deviation_strides[0],
        info.input_std_deviation_strides[1]
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
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = grad_input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = LayerNormBackwardInfo::createLayerNormBackwardInfo(
        grad_input_desc,
        grad_weight_desc,
        grad_bias_desc,
        grad_output_desc,
        weight_desc,
        input_standardization_desc,
        input_std_deviation_desc
    );
    CHECK_RESULT(result);
    const LayerNormBackwardInfo &info = result.take();
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
    const void * grad_output,
    const void * weight,
    const void * input_standardization,
    const void * input_std_deviation,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    #define CALCULATE_LAYER_NORM_BACKWARD(BLOCK_SIZE, TDATA) \
        calculate_layer_norm_backward<BLOCK_SIZE, TDATA>(_info, (TDATA *)grad_input, (TDATA *)grad_weight, (TDATA *)grad_bias, (const TDATA *)grad_output, (const TDATA *)weight, (const TDATA *)input_standardization, (const TDATA *)input_std_deviation, stream)
    #define CALCULATE_LAYER_NORM_BACKWARD_WITH_BLOCK_SIZE(BLOCK_SIZE)           \
    {                                                                           \
        if (_info.dtype == INFINI_DTYPE_F16)                                    \
            return CALCULATE_LAYER_NORM_BACKWARD(BLOCK_SIZE, half);             \
        else if (_info.dtype == INFINI_DTYPE_F32)                               \
            return CALCULATE_LAYER_NORM_BACKWARD(BLOCK_SIZE, float);            \
        else if (_info.dtype == INFINI_DTYPE_BF16)                              \
            return CALCULATE_LAYER_NORM_BACKWARD(BLOCK_SIZE, __nv_bfloat16);    \
        else                                                                    \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                              \
    }
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_LAYER_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_LAYER_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096)
        CALCULATE_LAYER_NORM_BACKWARD_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::layer_norm_backward::nvidia
