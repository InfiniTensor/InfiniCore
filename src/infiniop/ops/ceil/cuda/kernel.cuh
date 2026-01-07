#ifndef __CEIL_CUDA_H__
#define __CEIL_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cuda_fp16.h>

namespace op::ceil::cuda {
typedef struct CeilOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2ceil(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hceil(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(ceilf(x0), ceilf(x1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(ceilf(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return ceilf(x);
        } else if constexpr (std::is_integral_v<T>) {
            return x;
        } else {
            return std::ceil(x);
        }
    }
} CeilOp;
} // namespace op::ceil::cuda

#endif // __CEIL_CUDA_H__
