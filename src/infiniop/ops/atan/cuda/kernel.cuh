#ifndef __ATAN_CUDA_H__
#define __ATAN_CUDA_H__

#include <cmath>
#include <cuda_fp16.h>

namespace op::atan::cuda {
typedef struct AtanOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __floats2half2_rn(atanf(__half2float(__low2half(x))), atanf(__half2float(__high2half(x))));
        } else if constexpr (std::is_same_v<T, half>) {
            return __float2half(atanf(__half2float(x)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(atanf(x0), atanf(x1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(atanf(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return atanf(x);
        } else {
            return std::atan(x);
        }
    }
} AtanOp;
} // namespace op::atan::cuda

#endif // __ATAN_CUDA_H__
