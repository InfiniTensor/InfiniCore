#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace op::cuda {

// Inverse error function.
//
// We use a Winitzki-style approximation for an initial guess, then refine with
// a few Newton iterations. Starting with y=x converges poorly for x close to 1,
// which appears frequently in test inputs (torch.rand in [0,1)).
__device__ __forceinline__ float erfinv_impl(float x) {
    if (x == 1.0f) return CUDART_INF_F;
    if (x == -1.0f) return -CUDART_INF_F;
    if (x > 1.0f || x < -1.0f) return CUDART_NAN_F;
    if (x == 0.0f) return 0.0f;

    // Winitzki approximation (a = 0.147) for initial guess.
    // See: https://arxiv.org/abs/math/0306301 (and common implementations).
    const float a = 0.147f;
    const float ln = log1pf(-x * x); // ln(1 - x^2) <= 0
    const float t = 2.0f / (CUDART_PI_F * a) + ln * 0.5f;
    float inside = t * t - ln / a;
    inside = inside > 0.0f ? inside : 0.0f;
    float y0 = copysignf(sqrtf(sqrtf(inside) - t), x);

    // Fast path: a few Newton steps in float.
    // This is sufficient for most x and much faster than always refining in double.
    float y = y0;
    const float sqrt_pi_f = 1.7724538509055159f; // sqrt(pi)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float erf_y = erff(y);
        const float derf_dy = 2.0f / sqrt_pi_f * expf(-y * y);
        y = y - (erf_y - x) / derf_dy;
    }

    // Hybrid slow path: only for values extremely close to ±1 where float erf
    // quantization can cause Newton iterations to stagnate, leading to noticeable
    // absolute error in y (even if erff(y) == x in float).
    //
    // The threshold is chosen so the slow path is taken very rarely for typical
    // random inputs, minimizing warp divergence and preserving performance.
    const float ax = fabsf(x);
    if (1.0f - ax < 1e-4f) {
        const double xd = static_cast<double>(x);
        double yd = static_cast<double>(y);
        const double sqrt_pi = 1.7724538509055159; // sqrt(pi)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const double erf_y = erf(yd);
            const double derf_dy = 2.0 / sqrt_pi * exp(-yd * yd);
            yd = yd - (erf_y - xd) / derf_dy;
        }
        y = static_cast<float>(yd);
    }

    return y;
}

struct ErfinvOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            return erfinv_impl(x);
        } else if constexpr (std::is_same_v<T, double>) {
            // For double, use similar approach
            if (x == 1.0) return CUDART_INF;
            if (x == -1.0) return -CUDART_INF;
            if (x > 1.0 || x < -1.0) return CUDART_NAN;
            if (x == 0.0) return 0.0;
            const double a = 0.147;
            const double ln = log1p(-x * x);
            const double t = 2.0 / (CUDART_PI * a) + ln * 0.5;
            double inside = t * t - ln / a;
            inside = inside > 0.0 ? inside : 0.0;
            double y = copysign(sqrt(sqrt(inside) - t), x);

            const int max_iter = 30;
            const double tol = 1e-14;
            const double sqrt_pi = 1.7724538509055159;
            for (int i = 0; i < max_iter; ++i) {
                const double erf_y = erf(y);
                const double error = erf_y - x;
                if (fabs(error) < tol) break;
                const double derf_dy = 2.0 / sqrt_pi * exp(-y * y);
                y = y - error / derf_dy;
            }
            return y;
        } else {
            // For F16/BF16: promote to float, compute, then cast back
            float xf;
            if constexpr (std::is_same_v<T, half>) {
                xf = __half2float(x);
                return __float2half_rn(erfinv_impl(xf));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                xf = __bfloat162float(x);
                return __float2bfloat16_rn(erfinv_impl(xf));
            } else {
                xf = static_cast<float>(x);
                return static_cast<T>(erfinv_impl(xf));
            }
        }
    }
};

} // namespace op::cuda
