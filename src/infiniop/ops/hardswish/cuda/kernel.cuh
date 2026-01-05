#ifndef __HARDSWISH_CUDA_H__
#define __HARDSWISH_CUDA_H__

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace op::hardswish::cuda {

typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        // HardSwish 公式: x * min(max(x + 3, 0), 6) / 6

        // ----------------------------------------
        // Case 1: Half2 (FP16 向量化)
        // ----------------------------------------
        if constexpr (std::is_same_v<T, half2>) {
            // 常量定义
            const half2 three = __float2half2_rn(3.0f);
            const half2 zero = __float2half2_rn(0.0f);
            const half2 six = __float2half2_rn(6.0f);
            const half2 scale = __float2half2_rn(0.16666667f); // 1.0f / 6.0f

            // 1. x + 3
            half2 val = __hadd2(x, three);
            // 2. ReLU6: clamp(val, 0, 6) -> min(max(val, 0), 6)
            // 注意：__hmax2(val, zero) 相当于 ReLU
            //      __hmin2(..., six) 相当于 Cap at 6
#if __CUDA_ARCH__ >= 800
             // Ampere 架构及以上通常有更好的 hmin/hmax 支持
            val = __hmin2(__hmax2(val, zero), six);
#else
            // 旧架构兼容写法，虽然 intrinsics 一样，但逻辑清晰
            val = __hmax2(val, zero);
            val = __hmin2(val, six);
#endif
            // 3. x * val * 1/6
            return __hmul2(__hmul2(x, val), scale);

        } 
        // ----------------------------------------
        // Case 2: BF16 (BFloat16)
        // ----------------------------------------
        else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 目前 BF16 math intrinsics较少，通常转 float 计算
            const float x_f = __bfloat162float(x);
            // 计算 float 版本的 hardswish
            const float val = fminf(fmaxf(x_f + 3.0f, 0.0f), 6.0f);
            return __float2bfloat16(x_f * val * 0.16666667f);

        } 
        // ----------------------------------------
        // Case 3: Half (FP16 标量)
        // ----------------------------------------
        else if constexpr (std::is_same_v<T, half>) {
            const float x_f = __half2float(x);
            const float val = fminf(fmaxf(x_f + 3.0f, 0.0f), 6.0f);
            return __float2half(x_f * val * 0.16666667f);

        } 
        // ----------------------------------------
        // Case 4: Float (FP32)
        // ----------------------------------------
        else if constexpr (std::is_same_v<T, float>) {
            // fminf / fmaxf 会被编译为对应的 PTX 指令
            const float val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
            return x * val * 0.16666667f;

        } 
        // ----------------------------------------
        // Case 5: Double (FP64)
        // ----------------------------------------
        else if constexpr (std::is_same_v<T, double>) {
            const double val = fmin(fmax(x + 3.0, 0.0), 6.0);
            return x * val * (1.0 / 6.0);
        }
    }
} HardSwishOp;

} // namespace op::hardswish::cuda

#endif // __HARDSWISH_CUDA_H__