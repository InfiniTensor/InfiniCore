#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "layer_norm_nvidia.cuh"
#include "../../../../utils.h"
#include "../cuda/kernel.cuh"

// 支持混合数据类型的LayerNorm kernel
template <unsigned int BLOCK_SIZE, typename T, typename WeightT = T, typename BiasT = T>
__global__ void layerNormKernel(
    T *__restrict__ output,
    const T *__restrict__ input,
    const WeightT *__restrict__ weight,
    const BiasT *__restrict__ bias,
    T *__restrict__ input_std_deviation,
    T *__restrict__ input_standardization,
    size_t batch_size,
    size_t normalized_size,
    float eps,
    bool has_bias) {
    
    layerNormBlock<BLOCK_SIZE, T, WeightT, BiasT>(
        output, input, weight, bias,
        input_std_deviation, input_standardization,
        batch_size, normalized_size, eps, has_bias);
}

namespace op::layer_norm::nvidia {

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
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    float eps) {
    
    // 验证输入参数
    if (!handle_ || !desc_ptr || !output_desc || !input_desc || 
        !weight_desc || !input_std_deviation_desc || !input_standardization_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (eps <= 0.0f) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = input_desc->dtype();
    
    // 检查数据类型支持
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    // 检查数据类型一致性
    if (input_desc->dtype() != output_desc->dtype() ||
        input_desc->dtype() != input_std_deviation_desc->dtype() ||
        input_desc->dtype() != input_standardization_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查输入维度（至少3D）
    if (input_desc->ndim() < 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    // 检查形状兼容性
    if (input_desc->ndim() != output_desc->ndim() ||
        input_desc->ndim() != input_standardization_desc->ndim()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    for (size_t i = 0; i < input_desc->ndim(); ++i) {
        if (input_desc->dim(i) != output_desc->dim(i) ||
            input_desc->dim(i) != input_standardization_desc->dim(i)) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    // weight应该是1D张量，长度为最后一维
    size_t normalized_size = input_desc->dim(input_desc->ndim() - 1);
    if (weight_desc->ndim() != 1 || weight_desc->dim(0) != normalized_size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查bias形状（如果存在）
    bool has_bias = (bias_desc != nullptr);
    if (has_bias) {
        if (bias_desc->ndim() != 1 || bias_desc->dim(0) != normalized_size) {
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
    
    // 创建LayerNormInfo
    LayerNormInfo info;
    
    // 计算batch_size（除了最后一维的所有维度的乘积）
    info._batch_size = 1;
        for (size_t i = 0; i < input_desc->ndim() - 1; ++i) {
            info._batch_size *= input_desc->dim(i);
        }
        
        info._normalized_size = normalized_size;
    info.total_elements = input_desc->numel();
    info.input_size = input_desc->numel();
    info.output_size = output_desc->numel();
    info.dtype = dtype;
    info.eps = eps;
    info.has_bias = has_bias;
    
    // 复制形状和步长信息
    info.input_shape = input_desc->shape();
    info.output_shape = output_desc->shape();
    info.weight_shape = weight_desc->shape();
    if (has_bias) {
        info.bias_shape = bias_desc->shape();
    }
    info.input_std_deviation_shape = input_std_deviation_desc->shape();
    info.input_standardization_shape = input_standardization_desc->shape();
    
    info.input_strides = input_desc->strides();
    info.output_strides = output_desc->strides();
    info.weight_strides = weight_desc->strides();
    if (has_bias) {
        info.bias_strides = bias_desc->strides();
    }
    info.input_std_deviation_strides = input_std_deviation_desc->strides();
    info.input_standardization_strides = input_standardization_desc->strides();
    
    // LayerNorm不需要额外的workspace
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
infiniStatus_t launchKernel(
    const LayerNormInfo &info,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *input_std_deviation,
    void *input_standardization,
    cudaStream_t cuda_stream) {
    
    dim3 grid(info.batch_size());
    dim3 block(BLOCK_SIZE);
    
    switch (info.dtype) {
    case INFINI_DTYPE_F32:
        // 根据权重和偏置的实际数据类型进行正确的类型转换
        if (info.wtype == INFINI_DTYPE_F32) {
            // 权重是F32类型
            if (info.has_bias && info.btype == INFINI_DTYPE_F32) {
                // 偏置也是F32类型
                layerNormKernel<BLOCK_SIZE, float><<<grid, block, 0, cuda_stream>>>(
                    reinterpret_cast<float*>(output),
                    reinterpret_cast<const float*>(input),
                    reinterpret_cast<const float*>(weight),
                    reinterpret_cast<const float*>(bias),
                    reinterpret_cast<float*>(input_std_deviation),
                    reinterpret_cast<float*>(input_standardization),
                    info.batch_size(), info.dim(),
                    info.eps, info.has_bias);
            } else {
                // 偏置与输入类型相同或无偏置
                layerNormKernel<BLOCK_SIZE, float><<<grid, block, 0, cuda_stream>>>(
                    reinterpret_cast<float*>(output),
                    reinterpret_cast<const float*>(input),
                    reinterpret_cast<const float*>(weight),
                    reinterpret_cast<const float*>(bias),
                    reinterpret_cast<float*>(input_std_deviation),
                    reinterpret_cast<float*>(input_standardization),
                    info.batch_size(), info.dim(),
                    info.eps, info.has_bias);
            }
        } else {
            // 权重与输入类型相同
            layerNormKernel<BLOCK_SIZE, float><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<float*>(output),
                reinterpret_cast<const float*>(input),
                reinterpret_cast<const float*>(weight),
                reinterpret_cast<const float*>(bias),
                reinterpret_cast<float*>(input_std_deviation),
                reinterpret_cast<float*>(input_standardization),
                info.batch_size(), info.dim(),
                info.eps, info.has_bias);
        }
        break;
    case INFINI_DTYPE_F16:
        // 根据权重和偏置的实际数据类型进行正确的类型转换
         if (info.wtype == INFINI_DTYPE_F32) {
             // 权重是F32类型，需要特殊处理
             if (info.has_bias && info.btype == INFINI_DTYPE_F32) {
                 // 权重和偏置都是F32类型
                 layerNormKernel<BLOCK_SIZE, __half, float, float><<<grid, block, 0, cuda_stream>>>(
                     reinterpret_cast<__half*>(output),
                     reinterpret_cast<const __half*>(input),
                     reinterpret_cast<const float*>(weight),
                     reinterpret_cast<const float*>(bias),
                     reinterpret_cast<__half*>(input_std_deviation),
                     reinterpret_cast<__half*>(input_standardization),
                     info.batch_size(), info.dim(),
                     info.eps, info.has_bias);
             } else {
                 // 权重是F32，偏置与输入类型相同或无偏置
                 layerNormKernel<BLOCK_SIZE, __half, float, __half><<<grid, block, 0, cuda_stream>>>(
                     reinterpret_cast<__half*>(output),
                     reinterpret_cast<const __half*>(input),
                     reinterpret_cast<const float*>(weight),
                     reinterpret_cast<const __half*>(bias),
                     reinterpret_cast<__half*>(input_std_deviation),
                     reinterpret_cast<__half*>(input_standardization),
                     info.batch_size(), info.dim(),
                     info.eps, info.has_bias);
             }
         } else {
             // 权重与输入类型相同
             layerNormKernel<BLOCK_SIZE, __half><<<grid, block, 0, cuda_stream>>>(
                 reinterpret_cast<__half*>(output),
                 reinterpret_cast<const __half*>(input),
                 reinterpret_cast<const __half*>(weight),
                 reinterpret_cast<const __half*>(bias),
                 reinterpret_cast<__half*>(input_std_deviation),
                 reinterpret_cast<__half*>(input_standardization),
                 info.batch_size(), info.dim(),
                 info.eps, info.has_bias);
         }
        break;
    case INFINI_DTYPE_BF16:
        // 根据权重和偏置的实际数据类型进行正确的类型转换
         if (info.wtype == INFINI_DTYPE_F32) {
             // 权重是F32类型，需要特殊处理
             if (info.has_bias && info.btype == INFINI_DTYPE_F32) {
                 // 权重和偏置都是F32类型
                 layerNormKernel<BLOCK_SIZE, __nv_bfloat16, float, float><<<grid, block, 0, cuda_stream>>>(
                     reinterpret_cast<__nv_bfloat16*>(output),
                     reinterpret_cast<const __nv_bfloat16*>(input),
                     reinterpret_cast<const float*>(weight),
                     reinterpret_cast<const float*>(bias),
                     reinterpret_cast<__nv_bfloat16*>(input_std_deviation),
                     reinterpret_cast<__nv_bfloat16*>(input_standardization),
                     info.batch_size(), info.dim(),
                     info.eps, info.has_bias);
             } else {
                 // 权重是F32，偏置与输入类型相同或无偏置
                 layerNormKernel<BLOCK_SIZE, __nv_bfloat16, float, __nv_bfloat16><<<grid, block, 0, cuda_stream>>>(
                     reinterpret_cast<__nv_bfloat16*>(output),
                     reinterpret_cast<const __nv_bfloat16*>(input),
                     reinterpret_cast<const float*>(weight),
                     reinterpret_cast<const __nv_bfloat16*>(bias),
                     reinterpret_cast<__nv_bfloat16*>(input_std_deviation),
                     reinterpret_cast<__nv_bfloat16*>(input_standardization),
                     info.batch_size(), info.dim(),
                     info.eps, info.has_bias);
             }
         } else {
             // 权重与输入类型相同
             layerNormKernel<BLOCK_SIZE, __nv_bfloat16><<<grid, block, 0, cuda_stream>>>(
                 reinterpret_cast<__nv_bfloat16*>(output),
                 reinterpret_cast<const __nv_bfloat16*>(input),
                 reinterpret_cast<const __nv_bfloat16*>(weight),
                 reinterpret_cast<const __nv_bfloat16*>(bias),
                 reinterpret_cast<__nv_bfloat16*>(input_std_deviation),
                 reinterpret_cast<__nv_bfloat16*>(input_standardization),
                 info.batch_size(), info.dim(),
                 info.eps, info.has_bias);
         }
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
    void *input_std_deviation,
    void *input_standardization,
    void *stream) const {
    
    if (!output || !input || !weight || !input_std_deviation || !input_standardization) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (info.has_bias && !bias) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // 根据设备能力选择合适的block size
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            info, output, input, weight, bias,
            input_std_deviation, input_standardization, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            info, output, input, weight, bias,
            input_std_deviation, input_standardization, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            info, output, input, weight, bias,
            input_std_deviation, input_standardization, cuda_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::layer_norm::nvidia