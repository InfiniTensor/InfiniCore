#ifndef __SIN_CUDA_H__
#define __SIN_CUDA_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::sin::cuda {
typedef struct SinOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(input);
            float2 vr = make_float2(__sinf(vf.x), __sinf(vf.y));
            return __float22half2_rn(vr);
        } else if constexpr (std::is_same_v<T, half>) {
            float inputf = __half2float(input);
            return __float2half_rn(sinf(inputf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(input));
            float f1 = __bfloat162float(__high2bfloat16(input));
            return __floats2bfloat162_rn(__sinf(f0), __sinf(f1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float inputf = __bfloat162float(input);
            return __float2bfloat16_rn(__sinf(inputf));
        } else if constexpr (std::is_same_v<T, float>) {
            return sinf(input);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::sin(input);
        } else {
            return std::sin(input);
        }
    }
} SinOp;
} // namespace op::sin::cuda

#endif // __SIN_CUDA_H__
