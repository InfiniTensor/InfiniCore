#ifndef __LAYER_NORM_BACKWARD_KERNEL_CUH__
#define __LAYER_NORM_BACKWARD_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>

// GPU版本的高精度类型转换函数
template<typename T>
__device__ __forceinline__ T directCast(double val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(static_cast<float>(val));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(static_cast<float>(val));
    } else {
        return static_cast<T>(val);
    }
}

template<typename T>
__device__ __forceinline__ float preciseCast(const T& val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);
    } else {
        return static_cast<float>(val);
    }
}

// GPU版本的Kahan求和算法
__device__ __forceinline__ void kahanSum(float& sum, float& c, float val) {
    float y = val - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// LayerNorm反向传播kernel，支持混合数据类型
template <unsigned int BLOCK_SIZE, typename T, typename WeightT = T, typename BiasT = T>
__device__ void layerNormBackwardBlock(
    T *__restrict__ input_grad,
    WeightT *__restrict__ weight_grad,
    BiasT *__restrict__ bias_grad,
    const T *__restrict__ output_grad,
    const T *__restrict__ input,
    const WeightT *__restrict__ weight,
    const T *__restrict__ input_std_deviation,
    const T *__restrict__ input_standardization,
    size_t batch_size,
    size_t normalized_size,
    bool has_bias) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const T *grad_output_ptr = output_grad + batch_idx * normalized_size;
    const T *input_norm_ptr = input_standardization + batch_idx * normalized_size;
    T *grad_input_ptr = input_grad + batch_idx * normalized_size;
    
    float std_dev_val = preciseCast(input_std_deviation[batch_idx]);
    // 添加除零保护
    if (std_dev_val <= 1e-8f) {
        std_dev_val = 1e-8f;
    }
    float inv_std_dev = 1.0f / std_dev_val;
    
    // 使用shared memory来存储中间结果
    __shared__ float shared_sum_grad_out[BLOCK_SIZE];
    __shared__ float shared_sum_grad_out_norm[BLOCK_SIZE];
    
    // 计算中间变量用于grad_input
    float sum_grad_out = 0.0f, sum_grad_out_c = 0.0f;
    float sum_grad_out_norm = 0.0f, sum_grad_out_norm_c = 0.0f;
    
    for (int i = tid; i < normalized_size; i += BLOCK_SIZE) {
        float grad_out = preciseCast(grad_output_ptr[i]);
        float w_val;
        if constexpr (std::is_same_v<WeightT, float>) {
            w_val = weight[i];
        } else {
            w_val = preciseCast(weight[i]);
        }
        float input_norm_val = preciseCast(input_norm_ptr[i]);
        
        float grad_out_w = grad_out * w_val;
        kahanSum(sum_grad_out, sum_grad_out_c, grad_out_w);
        kahanSum(sum_grad_out_norm, sum_grad_out_norm_c, grad_out_w * input_norm_val);
    }
    
    shared_sum_grad_out[tid] = sum_grad_out;
    shared_sum_grad_out_norm[tid] = sum_grad_out_norm;
    __syncthreads();
    
    // 归约求和
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum_grad_out[tid] += shared_sum_grad_out[tid + stride];
            shared_sum_grad_out_norm[tid] += shared_sum_grad_out_norm[tid + stride];
        }
        __syncthreads();
    }
    
    float mean_grad_out = shared_sum_grad_out[0] / static_cast<float>(normalized_size);
    float mean_grad_out_norm = shared_sum_grad_out_norm[0] / static_cast<float>(normalized_size);
    
    // 计算grad_input
    for (int i = tid; i < normalized_size; i += BLOCK_SIZE) {
        float grad_out = preciseCast(grad_output_ptr[i]);
        float w_val;
        if constexpr (std::is_same_v<WeightT, float>) {
            w_val = weight[i];
        } else {
            w_val = preciseCast(weight[i]);
        }
        float input_norm_val = preciseCast(input_norm_ptr[i]);
        
        // LayerNorm反向传播公式: grad_input = inv_std * (grad_out * w - mean_grad_out - input_norm * mean_grad_out_norm)
        double grad_input_val = static_cast<double>(inv_std_dev) * 
            (grad_out * w_val - mean_grad_out - input_norm_val * mean_grad_out_norm);
        
        grad_input_ptr[i] = directCast<T>(grad_input_val);
    }
}

// 全局函数用于计算grad_weight和grad_bias
template <unsigned int BLOCK_SIZE, typename T, typename WeightT = T, typename BiasT = T>
__global__ void layerNormBackwardWeightBiasKernel(
    WeightT *__restrict__ weight_grad,
    BiasT *__restrict__ bias_grad,
    const T *__restrict__ output_grad,
    const T *__restrict__ input_standardization,
    size_t batch_size,
    size_t normalized_size,
    bool has_bias) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= normalized_size) return;
    
    // 使用Kahan求和算法累积grad_weight和grad_bias
    float grad_weight_sum = 0.0f, grad_weight_c = 0.0f;
    float grad_bias_sum = 0.0f, grad_bias_c = 0.0f;
    
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        float grad_out = preciseCast(output_grad[batch_idx * normalized_size + idx]);
        float input_norm_val = preciseCast(input_standardization[batch_idx * normalized_size + idx]);
        
        // 累积grad_weight
        kahanSum(grad_weight_sum, grad_weight_c, grad_out * input_norm_val);
        
        // 累积grad_bias
        if (has_bias) {
            kahanSum(grad_bias_sum, grad_bias_c, grad_out);
        }
    }
    
    // 写入grad_weight
    if constexpr (std::is_same_v<WeightT, float>) {
        weight_grad[idx] = grad_weight_sum;
    } else {
        weight_grad[idx] = directCast<WeightT>(static_cast<double>(grad_weight_sum));
    }
    
    // 写入grad_bias
    if (has_bias) {
        if constexpr (std::is_same_v<BiasT, float>) {
            bias_grad[idx] = grad_bias_sum;
        } else {
            bias_grad[idx] = directCast<BiasT>(static_cast<double>(grad_bias_sum));
        }
    }
}

#endif // __LAYER_NORM_BACKWARD_KERNEL_CUH__