#pragma once
#include "../batch_norm_backward.h"
#include <cub/cub.cuh>

namespace op::batch_norm_backward::metax {

// 设备端数据类型转换函数
template<typename T>
__device__ float device_cast_to_float(T val);

template<>
__device__ float device_cast_to_float<fp16_t>(fp16_t val) {
    // Convert custom fp16_t to __half first, then to float
    __half h_val;
    memcpy(&h_val, &val, sizeof(__half));
    return __half2float(h_val);
}

template<>
__device__ float device_cast_to_float<bf16_t>(bf16_t val) {
    // Convert custom bf16_t to __hpcc_bfloat16 first, then to float
    __hpcc_bfloat16 bf_val;
    memcpy(&bf_val, &val, sizeof(__hpcc_bfloat16));
    return __bfloat162float(bf_val);
}

template<>
__device__ float device_cast_to_float<__half>(__half val) {
    return __half2float(val);
}

template<>
__device__ float device_cast_to_float<__hpcc_bfloat16>(__hpcc_bfloat16 val) {
    return __bfloat162float(val);
}

template<>
__device__ float device_cast_to_float<float>(float val) {
    return val;
}

template<typename T>
__device__ T device_cast_from_float(float val);

template<>
__device__ fp16_t device_cast_from_float<fp16_t>(float val) {
    // Convert float to __half first, then to custom fp16_t
    __half h_val = __float2half(val);
    fp16_t result;
    memcpy(&result, &h_val, sizeof(fp16_t));
    return result;
}

template<>
__device__ bf16_t device_cast_from_float<bf16_t>(float val) {
    // Convert float to __hpcc_bfloat16 first, then to custom bf16_t
    __hpcc_bfloat16 bf_val = __float2bfloat16(val);
    bf16_t result;
    memcpy(&result, &bf_val, sizeof(bf16_t));
    return result;
}

template<>
__device__ __half device_cast_from_float<__half>(float val) {
    return __float2half(val);
}

template<>
__device__ __hpcc_bfloat16 device_cast_from_float<__hpcc_bfloat16>(float val) {
    return __float2bfloat16(val);
}

template<>
__device__ float device_cast_from_float<float>(float val) {
    return val;
}

// 优化的Kahan求和算法：使用float精度
__device__ inline void kahanSum(float& sum, float& c, float value) {
    float y = value - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// 针对低精度类型的数值稳定性函数
__device__ inline float sanitizeLowPrecision(float val, bool is_bf16) {
    // bf16/f16的数值范围：~±65504
    const float max_val = 65500.0f;
    const float min_val = -65500.0f;
    val = fmaxf(fminf(val, max_val), min_val);  // 裁剪到安全范围
    
    // 对bf16额外处理：避免尾数精度损失导致的NaN
    if (is_bf16 && (val != val)) {  // 检测NaN
        val = 0.0f;
    }
    return val;
}

// 通用数值清洗函数
__device__ inline float sanitizeValue(float val) {
    if (isnan(val) || isinf(val)) {
        return 0.0f;
    }
    return val;
}

/**
 * @brief BatchNorm backward kernel for MetaX devices
 * 
 * This kernel computes the backward pass for batch normalization, calculating
 * gradients for input (grad_input), weight (grad_weight), and bias (grad_bias).
 * 
 * @tparam BLOCK_SIZE Number of threads per block
 * @tparam T Data type for input/output tensors
 */
template <unsigned int BLOCK_SIZE, typename T>
__global__ void batchNormBackwardKernel(
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
    
    const size_t channel_idx = blockIdx.x;
    
    if (channel_idx >= channels) return;
    
    // 共享内存缓存当前通道的参数（仅加载1次，供块内所有线程使用）
    __shared__ float s_mean, s_var, s_weight;
    if (threadIdx.x == 0) {
        s_mean = device_cast_to_float(running_mean[channel_idx]);
        s_var = device_cast_to_float(running_var[channel_idx]);
        s_weight = device_cast_to_float(weight[channel_idx]);
    }
    __syncthreads();  // 等待共享内存加载完成
    
    // 使用共享内存中的参数，减少全局内存访问
    const float eps_val = eps;
    
    // 针对低精度类型优化的除零保护
    const float safe_var = fmaxf(s_var, 1e-12f);
    const float inv_std = 1.0f / sqrtf(safe_var + eps_val);
    
    // Step 1: 计算grad_bias (对grad_output求和)
    __shared__ float shared_bias_sum[BLOCK_SIZE];
    
    float bias_grad_sum = 0.0f;
    float bias_grad_c = 0.0f;
    
    // 优化内存访问：按线程ID连续访问空间维度（合并访问）
    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
            // 索引计算保证内存地址连续（对bf16/f16尤为重要）
            size_t idx = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx;
            float grad_out_val = device_cast_to_float(grad_output[idx]);
            kahanSum(bias_grad_sum, bias_grad_c, grad_out_val);
        }
    }
    
    shared_bias_sum[threadIdx.x] = bias_grad_sum;
    __syncthreads();
    
    // Block reduce for bias gradient
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_bias_sum[threadIdx.x] += shared_bias_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float block_bias_grad = shared_bias_sum[0];
    
    // Step 2: 计算grad_weight (grad_output * normalized_input的和)
    __shared__ float shared_weight_sum[BLOCK_SIZE];
    
    float weight_grad_sum = 0.0f;
    float weight_grad_c = 0.0f;
    
    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
            // 保持连续内存访问模式
            size_t idx = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx;
            float input_val = device_cast_to_float(input[idx]);
            float grad_out_val = device_cast_to_float(grad_output[idx]);
            float normalized = (input_val - s_mean) * inv_std;
            
            kahanSum(weight_grad_sum, weight_grad_c, grad_out_val * normalized);
        }
    }
    
    shared_weight_sum[threadIdx.x] = weight_grad_sum;
    __syncthreads();
    
    // Block reduce for weight gradient
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_weight_sum[threadIdx.x] += shared_weight_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float block_weight_grad = shared_weight_sum[0];
    
    // 存储grad_bias和grad_weight结果，应用低精度优化
    if (threadIdx.x == 0) {
        constexpr bool is_bf16 = std::is_same_v<T, __hpcc_bfloat16>;
        float sanitized_bias = sanitizeLowPrecision(sanitizeValue(block_bias_grad), is_bf16);
        float sanitized_weight = sanitizeLowPrecision(sanitizeValue(block_weight_grad), is_bf16);
        
        grad_bias[channel_idx] = device_cast_from_float<T>(sanitized_bias);
        grad_weight[channel_idx] = device_cast_from_float<T>(sanitized_weight);
    }
    
    // Step 3: 计算grad_input
    if (grad_input) {
        constexpr bool is_bf16 = std::is_same_v<T, __hpcc_bfloat16>;
        
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
                // 保持连续内存访问模式
                size_t idx = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx;
                
                // 先读取grad_output的值（处理inplace情况）
                float grad_out_val = device_cast_to_float(grad_output[idx]);
                
                // 在推理模式下，grad_input的计算简化为：
                // grad_input = grad_output * weight * inv_std
                float grad_in_val = grad_out_val * s_weight * inv_std;
                
                // 应用低精度数值稳定性处理
                grad_in_val = sanitizeLowPrecision(sanitizeValue(grad_in_val), is_bf16);
                grad_input[idx] = device_cast_from_float<T>(grad_in_val);
            }
        }
    }
}

} // namespace op::batch_norm_backward::metax