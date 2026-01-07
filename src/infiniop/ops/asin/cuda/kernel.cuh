#ifndef __ASIN_CUDA_H__
#define __ASIN_CUDA_H__

#include <cmath>
#include <cuda_fp16.h>

namespace op::asin::cuda {
typedef struct AsinOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __floats2half2_rn(asinf(__half2float(__low2half(x))), asinf(__half2float(__high2half(x))));
        } else if constexpr (std::is_same_v<T, half>) {
            return __float2half(asinf(__half2float(x)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(asinf(x0), asinf(x1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(asinf(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return asinf(x);
        } else {
            return std::asin(x);
        }
    }
} AsinOp;
} // namespace op::asin::cuda

#endif // __ASIN_CUDA_H__
