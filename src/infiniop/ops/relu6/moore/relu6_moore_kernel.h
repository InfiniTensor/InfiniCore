#ifndef __RELU6_MOORE_KERNEL_H__
#define __RELU6_MOORE_KERNEL_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <algorithm>

namespace op::relu6::moore {

typedef struct Relu6Op {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            return __floats2half2_rn(fminf(fmaxf(x0, 0.0f), 6.0f), fminf(fmaxf(x1, 0.0f), 6.0f));
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            return __float2half(fminf(fmaxf(xf, 0.0f), 6.0f));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(fminf(fmaxf(xf, 0.0f), 6.0f));
        } else if constexpr (std::is_same_v<T, float>) {
            return fminf(fmaxf(x, 0.0f), 6.0f);
        } else { // double
            return std::min(std::max(x, 0.0), 6.0);
        }
    }
} Relu6Op;

} // namespace op::relu6::moore

#endif // __RELU6_MOORE_KERNEL_H__
