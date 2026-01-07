#ifndef __COSH_CUDA_H__
#define __COSH_CUDA_H__

#include <cmath>
#include <cuda_fp16.h>

namespace op::cosh::cuda {
typedef struct CoshOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __floats2half2_rn(coshf(__half2float(__low2half(x))), coshf(__half2float(__high2half(x))));
        } else if constexpr (std::is_same_v<T, half>) {
            return __float2half(coshf(__half2float(x)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(coshf(x0), coshf(x1));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(coshf(__bfloat162float(x)));
        } else if constexpr (std::is_same_v<T, float>) {
            return coshf(x);
        } else {
            return std::cosh(x);
        }
    }
} CoshOp;
} // namespace op::cosh::cuda

#endif // __COSH_CUDA_H__
