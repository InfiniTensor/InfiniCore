#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "layer_norm_backward_nvidia.cuh"
#include "../../../../utils.h"
#include "../cuda/kernel.cuh"

// LayerNorm反向传播kernel，支持混合数据类型
template <unsigned int BLOCK_SIZE, typename T, typename WeightT = T, typename BiasT = T>
__global__ void layerNormBackwardKernel(
    T *__restrict__ grad_input,
    WeightT *__restrict__ grad_weight,
    BiasT *__restrict__ grad_bias,
    const T *__restrict__ grad_output,
    const T *__restrict__ input,
    const WeightT *__restrict__ weight,
    const T *__restrict__ input_std_deviation,
    const T *__restrict__ input_standardization,
    size_t batch_size,
    size_t normalized_size,
    bool has_bias) {
    
    layerNormBackwardBlock<BLOCK_SIZE, T, WeightT, BiasT>(
        grad_input, grad_weight, grad_bias, grad_output,
        input, weight, input_std_deviation, input_standardization,
        batch_size, normalized_size, has_bias);
}

namespace op::layer_norm_backward::nvidia {

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
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    float epsilon) {
    
    // 验证输入参数
    if (!handle_ || !desc_ptr || !grad_input_desc || !grad_weight_desc || 
        !grad_output_desc || !input_desc || !weight_desc || 
        !input_std_deviation_desc || !input_standardization_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (epsilon <= 0.0f) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = input_desc->dtype();
    
    // 检查数据类型支持
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    // 检查数据类型一致性
    if (input_desc->dtype() != grad_output_desc->dtype() ||
        input_desc->dtype() != grad_input_desc->dtype() ||
        input_desc->dtype() != input_std_deviation_desc->dtype() ||
        input_desc->dtype() != input_standardization_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查输入维度（至少3D）
    if (input_desc->ndim() < 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    // 检查形状兼容性
    if (input_desc->ndim() != grad_output_desc->ndim() ||
        input_desc->ndim() != grad_input_desc->ndim() ||
        input_desc->ndim() != input_standardization_desc->ndim()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    for (size_t i = 0; i < input_desc->ndim(); ++i) {
        if (input_desc->dim(i) != grad_output_desc->dim(i) ||
            input_desc->dim(i) != grad_input_desc->dim(i) ||
            input_desc->dim(i) != input_standardization_desc->dim(i)) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    // weight应该是1D张量，长度为最后一维
    size_t normalized_size = input_desc->dim(input_desc->ndim() - 1);
    if (weight_desc->ndim() != 1 || weight_desc->dim(0) != normalized_size ||
        grad_weight_desc->ndim() != 1 || grad_weight_desc->dim(0) != normalized_size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查bias形状（如果存在）
    bool has_bias = (grad_bias_desc != nullptr);
    if (has_bias) {
        if (grad_bias_desc->ndim() != 1 || grad_bias_desc->dim(0) != normalized_size) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    // 检查input_std_deviation形状（应该是input去掉最后一维）
    if (input_std_deviation_desc->ndim() != input_desc->ndim() - 1) {
        return INFINI_STATUS_BAD_PARAM;
    }
    for (size_t i = 0; i < input_std_deviation_desc->ndim(); ++i) {
        if (input_std_deviation_desc->dim(i) != input_desc->dim(i)) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    // 创建LayerNormBackwardInfo
    LayerNormBackwardInfo info;
    
    // 计算batch_size（除了最后一维的所有维度的乘积）
    info._batch_size = 1;
    for (size_t i = 0; i < input_desc->ndim() - 1; ++i) {
        info._batch_size *= input_desc->dim(i);
    }
    
    info._normalized_size = normalized_size;
    info.total_elements = input_desc->numel();
    info.input_size = input_desc->numel();
    info.output_size = grad_output_desc->numel();
    info.dtype = dtype;
    info.atype = dtype;
    info.wtype = weight_desc->dtype();
    info.btype = has_bias ? grad_bias_desc->dtype() : dtype;
    info.eps = epsilon;
    info.epsilon = epsilon;
    info.has_bias = has_bias;
    
    // 复制形状和步长信息
    info.input_grad_shape = grad_input_desc->shape();
    info.weight_grad_shape = grad_weight_desc->shape();
    if (has_bias) {
        info.bias_grad_shape = grad_bias_desc->shape();
    }
    info.output_grad_shape = grad_output_desc->shape();
    info.input_shape = input_desc->shape();
    info.weight_shape = weight_desc->shape();
    info.input_std_deviation_shape = input_std_deviation_desc->shape();
    info.input_standardization_shape = input_standardization_desc->shape();
    info.shape = grad_output_desc->shape();
    
    info.input_grad_strides = grad_input_desc->strides();
    info.weight_grad_strides = grad_weight_desc->strides();
    if (has_bias) {
        info.bias_grad_strides = grad_bias_desc->strides();
    }
    info.output_grad_strides = grad_output_desc->strides();
    info.input_strides = input_desc->strides();
    info.weight_strides = weight_desc->strides();
    info.input_std_deviation_strides = input_std_deviation_desc->strides();
    info.input_standardization_strides = input_standardization_desc->strides();
    
    // LayerNorm反向传播不需要额外的workspace
    size_t workspace_size = 0;
    
    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        std::move(info),
        workspace_size,
        handle->device,
        handle->device_id);
    
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchBackwardKernel(
    const LayerNormBackwardInfo &info,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    const void *input_std_deviation,
    const void *input_standardization,
    cudaStream_t cuda_stream) {
    
    dim3 grid_input(info.batch_size());
    dim3 block_input(BLOCK_SIZE);
    
    dim3 grid_weight((info.dim() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_weight(BLOCK_SIZE);
    
    switch (info.atype) {
    case INFINI_DTYPE_F32:
        // 启动input_grad计算kernel
        layerNormBackwardKernel<BLOCK_SIZE, float><<<grid_input, block_input, 0, cuda_stream>>>(
            reinterpret_cast<float*>(grad_input),
            reinterpret_cast<float*>(grad_weight),
            reinterpret_cast<float*>(grad_bias),
            reinterpret_cast<const float*>(grad_output),
            reinterpret_cast<const float*>(input),
            reinterpret_cast<const float*>(weight),
            reinterpret_cast<const float*>(input_std_deviation),
            reinterpret_cast<const float*>(input_standardization),
            info.batch_size(), info.dim(), info.has_bias);
        
        // 启动weight_grad和bias_grad计算kernel
        layerNormBackwardWeightBiasKernel<BLOCK_SIZE, float><<<grid_weight, block_weight, 0, cuda_stream>>>(
            reinterpret_cast<float*>(grad_weight),
            reinterpret_cast<float*>(grad_bias),
            reinterpret_cast<const float*>(grad_output),
            reinterpret_cast<const float*>(input_standardization),
            info.batch_size(), info.dim(), info.has_bias);
        break;
        
    case INFINI_DTYPE_F16:
        // 启动input_grad计算kernel
        layerNormBackwardKernel<BLOCK_SIZE, __half><<<grid_input, block_input, 0, cuda_stream>>>(
            reinterpret_cast<__half*>(grad_input),
            reinterpret_cast<__half*>(grad_weight),
            reinterpret_cast<__half*>(grad_bias),
            reinterpret_cast<const __half*>(grad_output),
            reinterpret_cast<const __half*>(input),
            reinterpret_cast<const __half*>(weight),
            reinterpret_cast<const __half*>(input_std_deviation),
            reinterpret_cast<const __half*>(input_standardization),
            info.batch_size(), info.dim(), info.has_bias);
        
        // 启动weight_grad和bias_grad计算kernel
        layerNormBackwardWeightBiasKernel<BLOCK_SIZE, __half><<<grid_weight, block_weight, 0, cuda_stream>>>(
            reinterpret_cast<__half*>(grad_weight),
            reinterpret_cast<__half*>(grad_bias),
            reinterpret_cast<const __half*>(grad_output),
            reinterpret_cast<const __half*>(input_standardization),
            info.batch_size(), info.dim(), info.has_bias);
        break;
        
    case INFINI_DTYPE_BF16:
        // 启动input_grad计算kernel
        layerNormBackwardKernel<BLOCK_SIZE, __nv_bfloat16><<<grid_input, block_input, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_input),
            reinterpret_cast<__nv_bfloat16*>(grad_weight),
            reinterpret_cast<__nv_bfloat16*>(grad_bias),
            reinterpret_cast<const __nv_bfloat16*>(grad_output),
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<const __nv_bfloat16*>(input_std_deviation),
            reinterpret_cast<const __nv_bfloat16*>(input_standardization),
            info.batch_size(), info.dim(), info.has_bias);
        
        // 启动weight_grad和bias_grad计算kernel
        layerNormBackwardWeightBiasKernel<BLOCK_SIZE, __nv_bfloat16><<<grid_weight, block_weight, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_weight),
            reinterpret_cast<__nv_bfloat16*>(grad_bias),
            reinterpret_cast<const __nv_bfloat16*>(grad_output),
            reinterpret_cast<const __nv_bfloat16*>(input_standardization),
            info.batch_size(), info.dim(), info.has_bias);
        break;
        
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::get_workspace_size(size_t *size) const {
    if (!size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    *size = _workspace_size;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    const void *input_std_deviation,
    const void *input_standardization,
    void *stream) const {
    
    if (!grad_input || !grad_weight || !grad_output || !input || 
        !weight || !input_std_deviation || !input_standardization) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (_info.has_bias && !grad_bias) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // 根据设备能力选择合适的block size
    if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchBackwardKernel<CUDA_BLOCK_SIZE_1024>(
            _info, grad_input, grad_weight, grad_bias, grad_output,
            input, weight, input_std_deviation, input_standardization, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchBackwardKernel<CUDA_BLOCK_SIZE_512>(
            _info, grad_input, grad_weight, grad_bias, grad_output,
            input, weight, input_std_deviation, input_standardization, cuda_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::layer_norm_backward::nvidia