#ifndef __FLOAT_POWER_CUDA_CUH__
#define __FLOAT_POWER_CUDA_CUH__

#if defined(__MACA__) || defined(__MACACC__)
    #include <maca_fp16.h>
    #include <maca_bfloat16.h>
    using nv_bfloat162 = __maca_bfloat162;
#else
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

#include <cmath>

namespace op::float_power::cuda {

// ==================================================================
// 基础定义: 向量化数据打包结构
// ==================================================================
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T val[N];
};

// ==================================================================
// Functor: 仅负责核心数学计算逻辑
// ==================================================================
struct FloatPowerFunctor {
    template <typename T_IN>
    __device__ __forceinline__ float compute(const T_IN &input, float exponent_val) const {
        // 将输入转为 float 参与计算，以保证计算精度和统一性
        float in_f = static_cast<float>(input);
        return powf(in_f, exponent_val);
    }
};

// ==================================================================
// 1. 通用处理 Kernel (Grid-Stride Loop)
// 用于处理不对齐、非向量化场景，支持 Tensor 指数
// ==================================================================
template <typename T_OUT, typename T_IN, typename T_EXP>
__global__ void float_power_kernel(
    T_OUT * __restrict__ output, 
    const T_IN * __restrict__ input, 
    const T_EXP * __restrict__ exponent, 
    float scalar_exponent, 
    bool is_scalar, 
    size_t numel, 
    FloatPowerFunctor functor) {
    
    // 使用网格跨度循环（Grid-Stride Loop）以掩盖访存延迟
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numel; 
         idx += blockDim.x * gridDim.x) {
         
        float exp_val_f = is_scalar ? scalar_exponent : static_cast<float>(exponent[idx]);
        output[idx] = static_cast<T_OUT>(functor.compute(input[idx], exp_val_f));
    }
}

// ==================================================================
// 2. 标量模式向量化 Kernel
// ==================================================================
template <typename T_OUT, typename T_IN, int PackSize>
__global__ void float_power_kernel_vectorized_scalar(
    T_OUT * __restrict__ output, 
    const T_IN * __restrict__ input, 
    float scalar_exponent, 
    size_t num_packs, 
    FloatPowerFunctor functor) {
    
    using PackTypeIn = Pack<T_IN, PackSize>;
    using PackTypeOut = Pack<T_OUT, PackSize>;

    auto in_vec = reinterpret_cast<const PackTypeIn*>(input);
    auto out_vec = reinterpret_cast<PackTypeOut*>(output);
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_packs) {
        PackTypeIn in_pack = in_vec[idx];
        PackTypeOut out_pack;
        
        #pragma unroll
        for (int i = 0; i < PackSize; ++i) {
            out_pack.val[i] = static_cast<T_OUT>(functor.compute(in_pack.val[i], scalar_exponent));
        }
        out_vec[idx] = out_pack;
    }
}

// ==================================================================
// 3. 张量模式向量化 Kernel
// 解决 Tensor 指数路径性能瓶颈的关键实现
// ==================================================================
template <typename T_OUT, typename T_IN, int PackSize>
__global__ void float_power_kernel_vectorized_tensor(
    T_OUT * __restrict__ output, 
    const T_IN * __restrict__ input, 
    const T_IN * __restrict__ exponent, 
    size_t num_packs, 
    FloatPowerFunctor functor) {
    
    using PackTypeIn = Pack<T_IN, PackSize>;
    using PackTypeOut = Pack<T_OUT, PackSize>;

    auto in_vec = reinterpret_cast<const PackTypeIn*>(input);
    auto exp_vec = reinterpret_cast<const PackTypeIn*>(exponent);
    auto out_vec = reinterpret_cast<PackTypeOut*>(output);
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_packs) {
        PackTypeIn in_pack = in_vec[idx];
        PackTypeIn exp_pack = exp_vec[idx]; 
        PackTypeOut out_pack;
        
        #pragma unroll
        for (int i = 0; i < PackSize; ++i) {
            float e = static_cast<float>(exp_pack.val[i]);
            // 已修正：使用 out_pack 而非未定义的 out_p
            out_pack.val[i] = static_cast<T_OUT>(functor.compute(in_pack.val[i], e));
        }
        out_vec[idx] = out_pack;
    }
}

} // namespace op::float_power::cuda

#endif // __FLOAT_POWER_CUDA_CUH__