#ifndef __ADAPTIVE_AVG_POOL1D_CUDA_H__
#define __ADAPTIVE_AVG_POOL1D_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>
#include <stdio.h>

namespace op::adaptive_avg_pool1d::cuda {

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ float to_float(const T &x) {
    if constexpr (std::is_same_v<T, half>) return __half2float(x);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, nv_bfloat16>) return __bfloat162float(x);
#endif
    else return static_cast<float>(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
    if constexpr (std::is_same_v<T, half>) return __float2half(x);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, nv_bfloat16>) return __float2bfloat16(x);
#endif
    else return static_cast<T>(x);
}


template <typename T>
__global__ void global_avg_pool1d_kernel(
    T* output,
    const T* input,
    size_t total_channels, // batch * channels
    size_t isize
) {
    // 每一个 Block 处理一个 (Batch, Channel) 任务
    size_t channel_idx = blockIdx.x; 
    if (channel_idx >= total_channels) return;

    const T* channel_input = input + channel_idx * isize;
    float sum = 0.0f;

    // Grid-Stride Loop within the channel (handle isize > blockDim.x)
    for (size_t i = threadIdx.x; i < isize; i += blockDim.x) {
        sum += to_float(channel_input[i]);
    }

    // Block 内归约
    // 1. Warp Reduce
    sum = warp_reduce_sum(sum);

    // 2. Shared Memory Reduce (跨 Warp)
    static __shared__ float shared_sum[32]; // Max 1024 threads / 32 = 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();


    if (wid == 0) {
        float val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0) {
            output[channel_idx] = from_float<T>(val / static_cast<float>(isize));
        }
    }
}


template <typename T>
__global__ void adaptive_avg_pool1d_general_kernel(
    T* output,
    const T* input,
    size_t batch_channels,
    size_t isize,
    size_t osize
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t total_elements = batch_channels * osize;

    // 预计算缩放因子，避免循环内除法
    float stride_factor = static_cast<float>(isize) / static_cast<float>(osize);

    for (; idx < total_elements; idx += stride) {
        size_t bc_idx = idx / osize;
        size_t out_idx = idx % osize;

        const T* in_ptr = input + bc_idx * isize;

        int istart = static_cast<int>(floorf(out_idx * stride_factor));
        int iend   = static_cast<int>(ceilf((out_idx + 1) * stride_factor));
        
        // 边界保护
        istart = max(0, istart);
        iend = min(static_cast<int>(isize), iend);

        float sum = 0.0f;
        int klen = iend - istart;

        
        for (int i = istart; i < iend; ++i) {
            sum += to_float(in_ptr[i]);
        }

        output[idx] = (klen > 0) ? from_float<T>(sum / klen) : from_float<T>(0.0f);
    }
}

// -------------------------------------------
// Launcher
// -------------------------------------------
template <typename T>
void launch_adaptive_avg_pool1d(
    T* output,
    const T* input,
    size_t batch_channels,
    size_t isize,
    size_t osize,
    cudaStream_t stream
) {
    // 策略分发
    if (osize == 1) {
        int threads = 256;
        // 如果 isize 很小，减少线程数
        if (isize < 256) threads = 128;
        if (isize < 128) threads = 64;
        if (isize < 64)  threads = 32;

        dim3 block(threads);
        dim3 grid(batch_channels); 
        
        global_avg_pool1d_kernel<T><<<grid, block, 0, stream>>>(
            output, input, batch_channels, isize
        );
    } else
        size_t total_output = batch_channels * osize;
        int threads = 256;
        int blocks = (total_output + threads - 1) / threads;
        
        
        if (blocks > 65535) blocks = 65535; 

        adaptive_avg_pool1d_general_kernel<T><<<blocks, threads, 0, stream>>>(
            output, input, batch_channels, isize, osize
        );
    }
}

} // namespace op::adaptive_avg_pool1d::cuda

#endif // __ADAPTIVE_AVG_POOL1D_CUDA_H__