#ifndef __BATCH_NORM_CUDA_KERNEL_CUH__
#define __BATCH_NORM_CUDA_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

template <unsigned int BLOCK_SIZE, typename T>
__device__ void batchNormBlock(
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
    
    const size_t total_elements = batch_size * spatial_size;
    const size_t channel_idx = blockIdx.x;
    
    if (channel_idx >= channels) return;
    
    // Step 1: Compute mean for this channel
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float sum = 0.0f;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
            size_t input_idx = batch_idx * channels * spatial_size + channel_idx * spatial_size + spatial_idx;
            sum += static_cast<float>(input[input_idx]);
        }
    }
    
    float block_mean = BlockReduce(temp_storage).Sum(sum) / static_cast<float>(total_elements);
    
    // Store mean in shared memory so all threads can access it
    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = block_mean;
    }
    __syncthreads();
    
    // Step 2: Compute variance for this channel using double precision for intermediate calculations
    __shared__ typename BlockReduce::TempStorage temp_storage2;
    double sum_sq_diff = 0.0;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
            size_t input_idx = batch_idx * channels * spatial_size + channel_idx * spatial_size + spatial_idx;
            double diff = static_cast<double>(input[input_idx]) - static_cast<double>(mean);
            sum_sq_diff += diff * diff;
        }
    }
    
    float block_variance = static_cast<float>(BlockReduce(temp_storage2).Sum(static_cast<float>(sum_sq_diff)) / static_cast<double>(total_elements));
    
    // Store variance in shared memory so all threads can access it
    __shared__ float variance;
    if (threadIdx.x == 0) {
        variance = block_variance;
        workspace_mean[channel_idx] = static_cast<T>(mean);
        workspace_var[channel_idx] = static_cast<T>(variance);
        
        // Update running statistics
        float old_mean = static_cast<float>(running_mean[channel_idx]);
        float old_var = static_cast<float>(running_var[channel_idx]);
        running_mean[channel_idx] = static_cast<T>((1.0f - momentum) * old_mean + momentum * mean);
        running_var[channel_idx] = static_cast<T>((1.0f - momentum) * old_var + momentum * variance);
        

    }
    
    __syncthreads();
    
    // Step 3: Normalize and apply affine transformation
    float std_inv = 1.0f / sqrtf(variance + eps);
    float w = static_cast<float>(weight[channel_idx]);
    float b = static_cast<float>(bias[channel_idx]);
    
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (size_t spatial_idx = threadIdx.x; spatial_idx < spatial_size; spatial_idx += BLOCK_SIZE) {
            size_t input_idx = batch_idx * channels * spatial_size + channel_idx * spatial_size + spatial_idx;
            
            float normalized = (static_cast<float>(input[input_idx]) - mean) * std_inv;
            output[input_idx] = static_cast<T>(normalized * w + b);
        }
    }
}

#endif // __BATCH_NORM_CUDA_KERNEL_CUH__