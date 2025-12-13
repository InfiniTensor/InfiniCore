#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

// Digamma function implementation
template <typename T>
__device__ __forceinline__ T digamma_impl(T x) {
    if (x <= 0.0f) return CUDART_NAN_F;
    
    T result = 0.0f;
    const T gamma = 0.57721566490153286060651209008240243104215933593992f;
    
    // Reduce to [1, 2] range
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
    
    // Series expansion
    T sum = 0.0f;
    for (int k = 1; k <= 20; ++k) {
        sum += x / (static_cast<T>(k) * (static_cast<T>(k) + x));
    }
    result += sum;
    
    return result;
}

template <typename T>
struct DigammaOp {
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            return digamma_impl(x);
        } else if constexpr (std::is_same_v<T, double>) {
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
        } else {
            // For F16/BF16: promote to float, compute, then cast back
            float xf = static_cast<float>(x);
            return static_cast<T>(digamma_impl(xf));
        }
    }
};

} // namespace op::cuda
