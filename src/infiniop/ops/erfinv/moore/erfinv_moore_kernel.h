#ifndef __ERFINV_MOORE_KERNEL_H__
#define __ERFINV_MOORE_KERNEL_H__

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace op::erfinv::moore {

// Inverse error function using Newton's method
template <typename T>
__device__ __forceinline__ T erfinv_impl(T x) {
    if (x >= 1.0f) return CUDART_INF_F;
    if (x <= -1.0f) return -CUDART_INF_F;
    if (x == 0.0f) return 0.0f;

    T y = x;
    const int max_iter = 10;
    const T tol = 1e-10f;
    const T sqrt_pi = 1.7724538509055159f;

    for (int i = 0; i < max_iter; ++i) {
        T erf_y = erff(y);
        T derf_dy = 2.0f / sqrt_pi * expf(-y * y);
        T error = erf_y - x;
        if (fabsf(error) < tol) break;
        y = y - error / derf_dy;
    }
    return y;
}

typedef struct ErfinvOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            return __floats2half2_rn(erfinv_impl(x0), erfinv_impl(x1));
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            return __float2half(erfinv_impl(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(erfinv_impl(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return erfinv_impl(x);
        } else { // double
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
        }
    }
} ErfinvOp;

} // namespace op::erfinv::moore

#endif // __ERFINV_MOORE_KERNEL_H__
