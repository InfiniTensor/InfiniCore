#ifndef __ROUND_CUDA_H__
#define __ROUND_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cuda_fp16.h>

namespace op::round::cuda {
typedef struct RoundOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2rint(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hrint(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(rintf(x0), rintf(x1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(rintf(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return rintf(x);
        } else if constexpr (std::is_integral_v<T>) {
            return x;
        } else {
            return std::nearbyint(x);
        }
    }
} RoundOp;
} // namespace op::round::cuda

#endif // __ROUND_CUDA_H__
