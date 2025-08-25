#include "../../../devices/nvidia/nvidia_common.cuh"
#include "batch_norm_backward_nvidia.cuh"
#include "../../../../utils.h"
#include "../cuda/kernel.cuh"

template <unsigned int BLOCK_SIZE, typename T>
INFINIOP_CUDA_KERNEL batchNormBackwardKernel(
    T *__restrict__ grad_input,
    T *__restrict__ grad_weight,
    T *__restrict__ grad_bias,
    const T *__restrict__ grad_output,
    const T *__restrict__ input,
    const T *__restrict__ weight,
    const T *__restrict__ running_mean,
    const T *__restrict__ running_var,
    size_t batch_size,
    size_t channels,
    size_t spatial_size,
    float momentum,
    float eps) {
    batchNormBackwardBlock<BLOCK_SIZE, T>(
        grad_input, grad_weight, grad_bias,
        grad_output, input, weight,
        running_mean, running_var,
        batch_size, channels, spatial_size,
        static_cast<double>(momentum), static_cast<double>(eps));
}

namespace op::batch_norm_backward::nvidia {

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
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    float momentum,
    float eps) {
    
    // 验证输入参数
    if (!handle_ || !desc_ptr || !grad_input_desc || !grad_weight_desc || 
        !grad_bias_desc || !grad_output_desc || !input_desc || !weight_desc ||
        !running_mean_desc || !running_var_desc) {
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
    if (input_desc->dtype() != grad_input_desc->dtype() ||
        input_desc->dtype() != grad_output_desc->dtype() ||
        input_desc->dtype() != weight_desc->dtype() ||
        input_desc->dtype() != grad_weight_desc->dtype() ||
        input_desc->dtype() != grad_bias_desc->dtype() ||
        input_desc->dtype() != running_mean_desc->dtype() ||
        input_desc->dtype() != running_var_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查形状兼容性
    if (input_desc->ndim() < 2) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 输入和梯度输出形状应该相同
    if (input_desc->ndim() != grad_output_desc->ndim() ||
        input_desc->ndim() != grad_input_desc->ndim()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    for (size_t i = 0; i < input_desc->ndim(); ++i) {
        if (input_desc->dim(i) != grad_output_desc->dim(i) ||
            input_desc->dim(i) != grad_input_desc->dim(i)) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    // weight, bias, running_mean, running_var 应该是1D张量，长度为channels
    size_t channels = input_desc->dim(1);
    if (weight_desc->ndim() != 1 || weight_desc->dim(0) != channels ||
        grad_weight_desc->ndim() != 1 || grad_weight_desc->dim(0) != channels ||
        grad_bias_desc->ndim() != 1 || grad_bias_desc->dim(0) != channels ||
        running_mean_desc->ndim() != 1 || running_mean_desc->dim(0) != channels ||
        running_var_desc->ndim() != 1 || running_var_desc->dim(0) != channels) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 创建BatchNormBackwardInfo
    auto result = BatchNormBackwardInfo::create(
        grad_input_desc, grad_weight_desc, grad_bias_desc,
        grad_output_desc, input_desc, weight_desc,
        running_mean_desc, running_var_desc, momentum, eps);
    CHECK_RESULT(result);
    auto info = result.take();
    
    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        std::move(info),
        0, // workspace_size will be calculated
        handle->device,
        handle->device_id
    );
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::get_workspace_size(
    size_t *workspace_size) const {
    
    if (!workspace_size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // GPU实现不需要额外的workspace
    *workspace_size = 0;
    
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename T>
static infiniStatus_t launchKernel(
    const BatchNormBackwardInfo &info,
    T *grad_input,
    T *grad_weight,
    T *grad_bias,
    const T *grad_output,
    const T *input,
    const T *weight,
    const T *running_mean,
    const T *running_var,
    cudaStream_t cuda_stream) {
    
    size_t batch_size = info._batch_size;
    size_t channels = info._channels;
    size_t spatial_size = info._spatial_size;
    double momentum = static_cast<double>(info.momentum);
    double eps = static_cast<double>(info.eps);
    
    // 每个channel使用一个block
    dim3 grid(channels);
    dim3 block(BLOCK_SIZE);
    
    batchNormBackwardKernel<BLOCK_SIZE, T><<<grid, block, 0, cuda_stream>>>(
        grad_input, grad_weight, grad_bias,
        grad_output, input, weight,
        running_mean, running_var,
        batch_size, channels, spatial_size,
        momentum, eps);
    
    CHECK_CUDA(cudaGetLastError());
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
    const void *running_mean,
    const void *running_var,
    void *stream) const {
    
    if (!grad_input || !grad_weight || !grad_bias || !grad_output || 
        !input || !weight || !running_mean || !running_var) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // 根据数据类型分发
    switch (info.dtype) {
        case INFINI_DTYPE_F32:
            // 根据设备能力选择合适的block size
            if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
                CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
                    info,
                    static_cast<float*>(grad_input),
                    static_cast<float*>(grad_weight),
                    static_cast<float*>(grad_bias),
                    static_cast<const float*>(grad_output),
                    static_cast<const float*>(input),
                    static_cast<const float*>(weight),
                    static_cast<const float*>(running_mean),
                    static_cast<const float*>(running_var),
                    cuda_stream));
            } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_512) {
                CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
                    info,
                    static_cast<float*>(grad_input),
                    static_cast<float*>(grad_weight),
                    static_cast<float*>(grad_bias),
                    static_cast<const float*>(grad_output),
                    static_cast<const float*>(input),
                    static_cast<const float*>(weight),
                    static_cast<const float*>(running_mean),
                    static_cast<const float*>(running_var),
                    cuda_stream));
            } else {
                return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
            }
            break;
        case INFINI_DTYPE_F16:
            if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
                CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
                    info,
                    static_cast<__half*>(grad_input),
                    static_cast<__half*>(grad_weight),
                    static_cast<__half*>(grad_bias),
                    static_cast<const __half*>(grad_output),
                    static_cast<const __half*>(input),
                    static_cast<const __half*>(weight),
                    static_cast<const __half*>(running_mean),
                    static_cast<const __half*>(running_var),
                    cuda_stream));
            } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_512) {
                CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
                    info,
                    static_cast<__half*>(grad_input),
                    static_cast<__half*>(grad_weight),
                    static_cast<__half*>(grad_bias),
                    static_cast<const __half*>(grad_output),
                    static_cast<const __half*>(input),
                    static_cast<const __half*>(weight),
                    static_cast<const __half*>(running_mean),
                    static_cast<const __half*>(running_var),
                    cuda_stream));
            } else {
                return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
            }
            break;
        case INFINI_DTYPE_BF16:
            if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
                CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
                    info,
                    static_cast<__nv_bfloat16*>(grad_input),
                    static_cast<__nv_bfloat16*>(grad_weight),
                    static_cast<__nv_bfloat16*>(grad_bias),
                    static_cast<const __nv_bfloat16*>(grad_output),
                    static_cast<const __nv_bfloat16*>(input),
                    static_cast<const __nv_bfloat16*>(weight),
                    static_cast<const __nv_bfloat16*>(running_mean),
                    static_cast<const __nv_bfloat16*>(running_var),
                    cuda_stream));
            } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_512) {
                CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
                    info,
                    static_cast<__nv_bfloat16*>(grad_input),
                    static_cast<__nv_bfloat16*>(grad_weight),
                    static_cast<__nv_bfloat16*>(grad_bias),
                    static_cast<const __nv_bfloat16*>(grad_output),
                    static_cast<const __nv_bfloat16*>(input),
                    static_cast<const __nv_bfloat16*>(weight),
                    static_cast<const __nv_bfloat16*>(running_mean),
                    static_cast<const __nv_bfloat16*>(running_var),
                    cuda_stream));
            } else {
                return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
            }
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::batch_norm_backward::nvidia