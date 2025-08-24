#include "../../../devices/nvidia/nvidia_common.cuh"
#include "batch_norm_nvidia.cuh"
#include "../../../../utils.h"
#include "../cuda/kernel.cuh"

template <unsigned int BLOCK_SIZE, typename T>
INFINIOP_CUDA_KERNEL batchNormKernel(
    T *__restrict__ output,
    const T *__restrict__ input,
    const T *__restrict__ weight,
    const T *__restrict__ bias,
    T *__restrict__ running_mean,
    T *__restrict__ running_var,
    T *__restrict__ workspace_mean,
    T *__restrict__ workspace_var,
    size_t batch_size,
    size_t channels,
    size_t spatial_size,
    float momentum,
    float eps) {
    batchNormBlock<BLOCK_SIZE, T>(
        output, input, weight, bias,
        running_mean, running_var,
        workspace_mean, workspace_var,
        batch_size, channels, spatial_size,
        momentum, eps);
}

namespace op::batch_norm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    float momentum,
    float eps) {
    
    // 验证输入参数
    if (!handle_ || !desc_ptr || !output_desc || !input_desc || 
        !weight_desc || !bias_desc || !running_mean_desc || !running_var_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (momentum < 0.0f || momentum > 1.0f || eps <= 0.0f) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = input_desc->dtype();
    
    // 检查数据类型支持
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    // 检查数据类型一致性
    if (input_desc->dtype() != output_desc->dtype() ||
        input_desc->dtype() != weight_desc->dtype() ||
        input_desc->dtype() != bias_desc->dtype() ||
        input_desc->dtype() != running_mean_desc->dtype() ||
        input_desc->dtype() != running_var_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查形状兼容性
    if (input_desc->ndim() < 2) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 输入和输出形状应该相同
    if (input_desc->ndim() != output_desc->ndim()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    for (size_t i = 0; i < input_desc->ndim(); ++i) {
        if (input_desc->dim(i) != output_desc->dim(i)) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    // weight, bias, running_mean, running_var 应该是1D张量，长度为channels
    size_t channels = input_desc->dim(1);
    if (weight_desc->ndim() != 1 || weight_desc->dim(0) != channels ||
        bias_desc->ndim() != 1 || bias_desc->dim(0) != channels ||
        running_mean_desc->ndim() != 1 || running_mean_desc->dim(0) != channels ||
        running_var_desc->ndim() != 1 || running_var_desc->dim(0) != channels) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 创建BatchNormInfo
    BatchNormInfo info;
    info.batch_size = input_desc->dim(0);
    info.channels = channels;
    
    // 计算spatial_size (除了batch和channel维度的所有维度的乘积)
    info.spatial_size = 1;
    for (size_t i = 2; i < input_desc->ndim(); ++i) {
        info.spatial_size *= input_desc->dim(i);
    }
    
    info.input_size = input_desc->numel();
    info.output_size = output_desc->numel();
    info.dtype = dtype;
    info.momentum = momentum;
    info.eps = eps;
    
    // 复制形状和步长信息
    info.input_shape = input_desc->shape();
    info.output_shape = output_desc->shape();
    info.input_strides = input_desc->strides();
    info.output_strides = output_desc->strides();
    info.weight_strides = weight_desc->strides();
    info.bias_strides = bias_desc->strides();
    info.running_mean_strides = running_mean_desc->strides();
    info.running_var_strides = running_var_desc->strides();
    
    // 计算workspace大小
    size_t dtype_size = infiniSizeOf(dtype);
    size_t workspace_size = 2 * channels * dtype_size; // mean + var
    
    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        std::move(info),
        workspace_size,
        handle_->device, handle_->device_id);
    
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    const BatchNormInfo &info,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *running_mean,
    void *running_var,
    void *workspace,
    cudaStream_t cuda_stream) {
    
    size_t dtype_size = infiniSizeOf(info.dtype);
    void *workspace_mean = workspace;
    void *workspace_var = static_cast<char*>(workspace) + info.channels * dtype_size;
    
    dim3 grid(info.channels);
    dim3 block(BLOCK_SIZE);
    
    switch (info.dtype) {
    case INFINI_DTYPE_F16:
        batchNormKernel<BLOCK_SIZE, half><<<grid, block, 0, cuda_stream>>>(
            reinterpret_cast<half*>(output),
            reinterpret_cast<const half*>(input),
            reinterpret_cast<const half*>(weight),
            reinterpret_cast<const half*>(bias),
            reinterpret_cast<half*>(running_mean),
            reinterpret_cast<half*>(running_var),
            reinterpret_cast<half*>(workspace_mean),
            reinterpret_cast<half*>(workspace_var),
            info.batch_size, info.channels, info.spatial_size,
            info.momentum, info.eps);
        break;
    case INFINI_DTYPE_F32:
        batchNormKernel<BLOCK_SIZE, float><<<grid, block, 0, cuda_stream>>>(
            reinterpret_cast<float*>(output),
            reinterpret_cast<const float*>(input),
            reinterpret_cast<const float*>(weight),
            reinterpret_cast<const float*>(bias),
            reinterpret_cast<float*>(running_mean),
            reinterpret_cast<float*>(running_var),
            reinterpret_cast<float*>(workspace_mean),
            reinterpret_cast<float*>(workspace_var),
            info.batch_size, info.channels, info.spatial_size,
            info.momentum, info.eps);
        break;
    case INFINI_DTYPE_BF16:
        batchNormKernel<BLOCK_SIZE, __nv_bfloat16><<<grid, block, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output),
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<const __nv_bfloat16*>(bias),
            reinterpret_cast<__nv_bfloat16*>(running_mean),
            reinterpret_cast<__nv_bfloat16*>(running_var),
            reinterpret_cast<__nv_bfloat16*>(workspace_mean),
            reinterpret_cast<__nv_bfloat16*>(workspace_var),
            info.batch_size, info.channels, info.spatial_size,
            info.momentum, info.eps);
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
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *running_mean,
    void *running_var,
    void *stream) const {
    
    if (!output || !input || !weight || !bias || !running_mean || !running_var) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // 根据设备能力选择合适的block size
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            info, output, input, weight, bias,
            running_mean, running_var, workspace, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            info, output, input, weight, bias,
            running_mean, running_var, workspace, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            info, output, input, weight, bias,
            running_mean, running_var, workspace, cuda_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::batch_norm::nvidia