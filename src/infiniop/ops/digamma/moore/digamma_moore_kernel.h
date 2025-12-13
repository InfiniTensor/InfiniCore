#ifndef __DIGAMMA_MOORE_KERNEL_H__
#define __DIGAMMA_MOORE_KERNEL_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace op::digamma::moore {

template <typename T>
__device__ __forceinline__ T digamma_impl(T x) {
    if (x <= 0.0f) return CUDART_NAN_F;
    
    T result = 0.0f;
    const T gamma = 0.57721566490153286060651209008240243104215933593992f;
    
    while (x < 1.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    while (x > 2.0f) {
        x -= 1.0f;
        result += 1.0f / x;
    }
    
    result -= gamma;
    result -= 1.0f / x;
    
    T sum = 0.0f;
    for (int k = 1; k <= 20; ++k) {
        sum += x / (static_cast<T>(k) * (static_cast<T>(k) + x));
    }
    result += sum;
    
    return result;
}

typedef struct DigammaOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            return __floats2half2_rn(digamma_impl(x0), digamma_impl(x1));
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            return __float2half(digamma_impl(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(digamma_impl(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return digamma_impl(x);
        } else { // double
            if (x <= 0.0) return CUDART_NAN;
            double result = 0.0;
            const double gamma = 0.57721566490153286060651209008240243104215933593992;
            while (x < 1.0) {
                result -= 1.0 / x;
                x += 1.0;
            }
            while (x > 2.0) {
                x -= 1.0;
                result += 1.0 / x;
            }
            result -= gamma;
            result -= 1.0 / x;
            double sum = 0.0;
            for (int k = 1; k <= 20; ++k) {
                sum += x / (static_cast<double>(k) * (static_cast<double>(k) + x));
            }
            result += sum;
            return result;
        }
    }
} DigammaOp;

} // namespace op::digamma::moore

#endif // __DIGAMMA_MOORE_KERNEL_H__
