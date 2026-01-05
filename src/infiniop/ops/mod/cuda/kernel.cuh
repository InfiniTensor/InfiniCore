#ifndef __MOD_CUDA_H__
#define __MOD_CUDA_H__

#include <cmath>
#include <cuda_fp16.h>

namespace op::mod::cuda {
typedef struct ModOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 a_f2 = __half22float2(a);
            float2 b_f2 = __half22float2(b);
            return __float22half2_rn(make_float2(std::fmod(a_f2.x, b_f2.x), std::fmod(a_f2.y, b_f2.y)));
        } else if constexpr (std::is_same_v<T, half>) {
            float a_ = __half2float(a);
            float b_ = __half2float(b);
            return __float2half(std::fmod(a_, b_));
        } else if constexpr (std::is_floating_point_v<T>) {
            return std::fmod(a, b);
        } else {
            return a % b;
        }
    }
} ModOp;
} // namespace op::mod::cuda

#endif // __MOD_CUDA_H__
