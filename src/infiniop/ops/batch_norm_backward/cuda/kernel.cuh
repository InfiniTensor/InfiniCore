#ifndef __BATCH_NORM_BACKWARD_CUDA_KERNEL_CUH__
#define __BATCH_NORM_BACKWARD_CUDA_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// 优化的类型转换函数：使用float作为中间类型，减少转换开销
template<typename T>
__device__ inline float preciseCast(const T& val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);  // f16->float（硬件原生支持）
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);  // bf16->float（硬件原生支持）
    } else {
        return static_cast<float>(val);
    }
}

template<typename T>
__device__ inline T ultraPreciseCast(float val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(val);
    } else {
        return static_cast<T>(val);
    }
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

// BatchNorm反向传播kernel，支持混合数据类型
template <unsigned int BLOCK_SIZE, typename T>
__device__ void batchNormBackwardBlock(
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
    double momentum,
    double eps) {
    
    const size_t channel_idx = blockIdx.x;
    
    if (channel_idx >= channels) return;
    
    // 共享内存缓存当前通道的参数（仅加载1次，供块内所有线程使用）
    __shared__ float s_mean, s_var, s_weight;
    if (threadIdx.x == 0) {
        s_mean = preciseCast(running_mean[channel_idx]);
        s_var = preciseCast(running_var[channel_idx]);
        s_weight = preciseCast(weight[channel_idx]);
    }
    __syncthreads();  // 等待共享内存加载完成
    
    // 使用共享内存中的参数，减少全局内存访问
    const float eps_val = static_cast<float>(eps);
    
    // 针对低精度类型优化的除零保护
    const float safe_var = fmaxf(s_var, 1e-12f);
    const float inv_std = 1.0f / sqrtf(safe_var + eps_val);
    
    // Step 1: 计算grad_bias (对grad_output求和)
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_bias;
    
    float bias_grad_sum = 0.0f;
    float bias_grad_c = 0.0f;
    
    // 优化内存访问：按线程ID连续访问空间维度（合并访问）
    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
            // 索引计算保证内存地址连续（对bf16/f16尤为重要）
            size_t idx = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx;
            float grad_out_val = preciseCast(grad_output[idx]);
            kahanSum(bias_grad_sum, bias_grad_c, grad_out_val);
        }
    }
    
    float block_bias_grad = BlockReduce(temp_storage_bias).Sum(bias_grad_sum);
    
    // Step 2: 计算grad_weight (grad_output * normalized_input的和)
    // 使用double精度进行累积以提高BF16/F16的计算精度
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduceDouble;
    __shared__ typename BlockReduceDouble::TempStorage temp_storage_weight;
    
    double weight_grad_sum = 0.0;
    double weight_grad_c = 0.0;
    
    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
            // 保持连续内存访问模式
            size_t idx = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx;
            double input_val = static_cast<double>(preciseCast(input[idx]));
            double grad_out_val = static_cast<double>(preciseCast(grad_output[idx]));
            double normalized = (input_val - static_cast<double>(s_mean)) * static_cast<double>(inv_std);
            
            // 使用double精度的Kahan求和
            double y = grad_out_val * normalized - weight_grad_c;
            double t = weight_grad_sum + y;
            weight_grad_c = (t - weight_grad_sum) - y;
            weight_grad_sum = t;
        }
    }
    
    double block_weight_grad = BlockReduceDouble(temp_storage_weight).Sum(weight_grad_sum);
    
    // 存储grad_bias和grad_weight结果，应用低精度优化
    if (threadIdx.x == 0) {
        constexpr bool is_bf16 = std::is_same_v<T, __nv_bfloat16>;
        float sanitized_bias = sanitizeLowPrecision(sanitizeValue(block_bias_grad), is_bf16);
        // 将double精度的grad_weight转换为float再进行数值稳定性处理
        float weight_grad_float = static_cast<float>(block_weight_grad);
        float sanitized_weight = sanitizeLowPrecision(sanitizeValue(weight_grad_float), is_bf16);
        
        grad_bias[channel_idx] = ultraPreciseCast<T>(sanitized_bias);
        grad_weight[channel_idx] = ultraPreciseCast<T>(sanitized_weight);
    }
    
    // Step 3: 计算grad_input
    if (grad_input) {
        constexpr bool is_bf16 = std::is_same_v<T, __nv_bfloat16>;
        
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
                // 保持连续内存访问模式
                size_t idx = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx;
                
                // 先读取grad_output的值（处理inplace情况）
                float grad_out_val = preciseCast(grad_output[idx]);
                
                // 在推理模式下，grad_input的计算简化为：
                // grad_input = grad_output * weight * inv_std
                float grad_in_val = grad_out_val * s_weight * inv_std;
                
                // 应用低精度数值稳定性处理
                grad_in_val = sanitizeLowPrecision(sanitizeValue(grad_in_val), is_bf16);
                grad_input[idx] = ultraPreciseCast<T>(grad_in_val);
            }
        }
    }
}

#endif // __BATCH_NORM_BACKWARD_CUDA_KERNEL_CUH__