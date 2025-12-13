#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

// Inverse error function using Newton's method
template <typename T>
__device__ __forceinline__ T erfinv_impl(T x) {
    if (x >= 1.0f) return CUDART_INF_F;
    if (x <= -1.0f) return -CUDART_INF_F;
    if (x == 0.0f) return 0.0f;

    T y = x; // Initial guess
    const int max_iter = 10;
    const T tol = static_cast<T>(1e-10f);
    const T sqrt_pi = 1.7724538509055159f; // sqrt(pi)

    for (int i = 0; i < max_iter; ++i) {
        T erf_y = erff(y);
        T derf_dy = 2.0f / sqrt_pi * expf(-y * y);
        T error = erf_y - x;
        if (fabsf(error) < tol) break;
        y = y - error / derf_dy;
    }
    return y;
}

template <typename T>
struct ErfinvOp {
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            return erfinv_impl(x);
        } else if constexpr (std::is_same_v<T, double>) {
            // For double, use similar approach
            if (x >= 1.0) return CUDART_INF;
            if (x <= -1.0) return -CUDART_INF;
            if (x == 0.0) return 0.0;
            double y = x;
            const int max_iter = 10;
            const double tol = 1e-10;
            const double sqrt_pi = 1.7724538509055159;
            for (int i = 0; i < max_iter; ++i) {
                double erf_y = erf(y);
                double derf_dy = 2.0 / sqrt_pi * exp(-y * y);
                double error = erf_y - x;
                if (fabs(error) < tol) break;
                y = y - error / derf_dy;
            }
            return y;
        } else {
            // For F16/BF16: promote to float, compute, then cast back
            float xf = static_cast<float>(x);
            return static_cast<T>(erfinv_impl(xf));
        }
    }
};

} // namespace op::cuda
