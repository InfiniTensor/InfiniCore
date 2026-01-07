#ifndef __POW_CUDA_H__
#define __POW_CUDA_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::pow::cuda {
typedef struct PowOp {
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 a_f2 = __half22float2(a);
            float2 b_f2 = __half22float2(b);
            return __float22half2_rn(make_float2(__powf(a_f2.x, b_f2.x), __powf(a_f2.y, b_f2.y)));
        } else if constexpr (std::is_same_v<T, half>) {
            float a_ = __half2float(a);
            float b_ = __half2float(b);
            float ans_f = __powf(a_, b_);
            return __float2half(isnan(ans_f) ? std::pow(a_, b_) : ans_f);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float2 a_f2 = __bfloat1622float2(a);
            float2 b_f2 = __bfloat1622float2(b);
            return __floats2bfloat162_rn(__powf(a_f2.x, b_f2.x), __powf(a_f2.y, b_f2.y));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float a_ = __bfloat162float(a);
            float b_ = __bfloat162float(b);
            return __float2bfloat16_rn(__powf(a_, b_));
        } else if constexpr (std::is_same_v<T, float>) {
            return __powf(a, b);
        } else {
            return std::pow(a, b);
        }
    }
} PowOp;

} // namespace op::pow::cuda

#endif // __POW_CUDA_H__
