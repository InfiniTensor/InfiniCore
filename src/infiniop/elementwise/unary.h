#ifndef __INFINIOP_ELEMENTWISE_UNARY_H__
#define __INFINIOP_ELEMENTWISE_UNARY_H__

#include <algorithm>
#include <cmath>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// Include device-specific type aliases for cuda_bfloat16
#include "../devices/nvidia/nvidia_kernel_common.cuh"
#endif

namespace op::elementwise::unary {

/**
 * @brief Represents all the currently defined unary operations.
 *
 * This enum is used to specify which unary operation to perform
 * in the generic UnaryOp template.
 */
enum class UnaryMode {
    // Math operations:
    Abs,
    Exp,
    Exp2, // exp2: 2^x
    Log,
    Log2,  // log2: log base 2
    Log10, // log10: log base 10
    Log1p, // log1p: log(1 + x), numerically stable for values close to zero
    Reciprocal,
    Sqrt,
    Square,
    Rsqrt,
    Neg,
    Ceil,
    Floor,
    Round,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Relu,
    Sigmoid,
    Sign,
    Erf,
    Hardswish,
    IsNan,
    IsInf,
    IsFinite,
    Sinc,
};

// Helper template for static_assert in else branches
template <UnaryMode M>
struct always_false : std::false_type {};

/**
 * @brief Generic unary operation template that performs different operations
 *        based on the specified UnaryMode.
 *
 * This template allows multiple unary operators (abs, log, sin, cos, etc.)
 * to share the same implementation infrastructure while only differing in the
 * operation mode.
 *
 * @tparam Mode The unary operation mode (from UnaryMode enum)
 */
template <UnaryMode Mode>
struct UnaryOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        if constexpr (Mode == UnaryMode::Abs) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fabs(x);
            } else {
                return std::abs(x);
            }
        } else if constexpr (Mode == UnaryMode::Exp) {
            return std::exp(x);
        } else if constexpr (Mode == UnaryMode::Exp2) {
            // exp2: 2^x
            return std::exp2(x);
        } else if constexpr (Mode == UnaryMode::Log) {
            return std::log(x);
        } else if constexpr (Mode == UnaryMode::Log2) {
            // log2: log base 2
            return std::log2(x);
        } else if constexpr (Mode == UnaryMode::Log10) {
            // log10: log base 10
            return std::log10(x);
        } else if constexpr (Mode == UnaryMode::Log1p) {
            // log1p: log(1 + x), numerically stable for values close to zero
            return std::log1p(x);
        } else if constexpr (Mode == UnaryMode::Reciprocal) {
            return T(1) / x;
        } else if constexpr (Mode == UnaryMode::Sqrt) {
            return std::sqrt(x);
        } else if constexpr (Mode == UnaryMode::Square) {
            return x * x;
        } else if constexpr (Mode == UnaryMode::Rsqrt) {
            return T(1) / std::sqrt(x);
        } else if constexpr (Mode == UnaryMode::Neg) {
            return -x;
        } else if constexpr (Mode == UnaryMode::Ceil) {
            return std::ceil(x);
        } else if constexpr (Mode == UnaryMode::Floor) {
            return std::floor(x);
        } else if constexpr (Mode == UnaryMode::Round) {
            if constexpr (std::is_integral_v<T>) {
                return x;
            } else {
                return std::nearbyint(x);
            }
        } else if constexpr (Mode == UnaryMode::Sin) {
            return std::sin(x);
        } else if constexpr (Mode == UnaryMode::Cos) {
            return std::cos(x);
        } else if constexpr (Mode == UnaryMode::Tan) {
            return std::tan(x);
        } else if constexpr (Mode == UnaryMode::Asin) {
            return std::asin(x);
        } else if constexpr (Mode == UnaryMode::Acos) {
            return std::acos(x);
        } else if constexpr (Mode == UnaryMode::Atan) {
            return std::atan(x);
        } else if constexpr (Mode == UnaryMode::Sinh) {
            return std::sinh(x);
        } else if constexpr (Mode == UnaryMode::Cosh) {
            return std::cosh(x);
        } else if constexpr (Mode == UnaryMode::Tanh) {
            return std::tanh(x);
        } else if constexpr (Mode == UnaryMode::Asinh) {
            return std::asinh(x);
        } else if constexpr (Mode == UnaryMode::Acosh) {
            return std::acosh(x);
        } else if constexpr (Mode == UnaryMode::Atanh) {
            return std::atanh(x);
        } else if constexpr (Mode == UnaryMode::Relu) {
            return x > T(0) ? x : T(0);
        } else if constexpr (Mode == UnaryMode::Sigmoid) {
            return T(1) / (T(1) + std::exp(-x));
        } else if constexpr (Mode == UnaryMode::Sign) {
            return x > T(0) ? T(1) : (x == T(0) ? T(0) : T(-1));
        } else if constexpr (Mode == UnaryMode::Erf) {
            return std::erf(x);
        } else if constexpr (Mode == UnaryMode::IsNan) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::isnan(x) ? T(1) : T(0);
            } else {
                // For integral types, NaN doesn't exist, so always return 0
                return T(0);
            }
        } else if constexpr (Mode == UnaryMode::IsInf) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::isinf(x) ? T(1) : T(0);
            } else {
                // For integral types, Inf doesn't exist, so always return 0
                return T(0);
            }
        } else if constexpr (Mode == UnaryMode::IsFinite) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::isfinite(x) ? T(1) : T(0);
            } else {
                // For integral types, all values are finite, so always return 1
                return T(1);
            }
        } else if constexpr (Mode == UnaryMode::Sinc) {
            // sinc(x) = sin(x) / x, sinc(0) = 1
            // For small values, use Taylor expansion for numerical stability
            // sinc(x) ≈ 1 - x²/6 + x⁴/120 - x⁶/5040
            if constexpr (std::is_floating_point_v<T>) {
                T abs_x = std::abs(x);
                if (abs_x < T(1e-2)) {
                    T x2 = x * x;
                    return T(1) - x2 * (T(1) / T(6) - x2 * (T(1) / T(120) - x2 * (T(1) / T(5040))));
                } else {
                    return std::sin(x) / x;
                }
            } else {
                // For integral types, sinc is not well-defined, return 1 for 0, 0 otherwise
                return x == T(0) ? T(1) : T(0);
            }
        } else if constexpr (Mode == UnaryMode::Hardswish) {
            if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(0);
            } else {
                // x * clamp(x + 3, 0, 6) / 6
                // Use template type T directly instead of double for better performance
                T y = x + T(3);
                y = std::min(std::max(y, T(0)), T(6));
                return x * (y / T(6));
            }
        } else {
            static_assert(always_false<Mode>::value, "Unsupported unary operation mode");
            return x;
        }
    }
};

#ifdef __CUDACC__
/**
 * @brief CUDA-specific unary operation template that performs different operations
 *        based on the specified UnaryMode, using CUDA-optimized functions.
 *
 * This template provides CUDA device functions optimized for GPU execution,
 * using intrinsics like __habs2, __logf, __sinf, etc.
 *
 * @tparam Mode The unary operation mode (from UnaryMode enum)
 */
namespace cuda {
template <UnaryMode Mode>
struct UnaryOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (Mode == UnaryMode::Abs) {
            if constexpr (std::is_same_v<T, half2>) {
                return __habs2(x);
            } else if constexpr (std::is_same_v<T, half>) {
                return __habs(x);
            } else if constexpr (std::is_floating_point_v<T>) {
                return std::fabs(x);
            } else {
                return std::abs(x);
            }
        } else if constexpr (Mode == UnaryMode::Exp) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(__expf(x_f2.x), __expf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(__expf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 x_f2 = __bfloat1622float2(x);
                return __floats2bfloat162_rn(__expf(x_f2.x), __expf(x_f2.y));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(__expf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return __expf(x);
            } else {
                return std::exp(x);
            }
        } else if constexpr (Mode == UnaryMode::Exp2) {
            // exp2: 2^x
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(exp2f(x_f2.x), exp2f(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(exp2f(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 x_f2 = __bfloat1622float2(x);
                return __floats2bfloat162_rn(exp2f(x_f2.x), exp2f(x_f2.y));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(exp2f(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return exp2f(x);
            } else {
                return std::exp2(x);
            }
        } else if constexpr (Mode == UnaryMode::Log) {
            if constexpr (std::is_same_v<T, half2>) {
                return h2log(x);
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(__logf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(logf(x0), logf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(logf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return __logf(x);
            } else {
                return std::log(x);
            }
        } else if constexpr (Mode == UnaryMode::Log2) {
            // log2: log base 2
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(log2f(x_f2.x), log2f(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(log2f(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(log2f(x0), log2f(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(log2f(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return log2f(x);
            } else {
                return std::log2(x);
            }
        } else if constexpr (Mode == UnaryMode::Log10) {
            // log10: log base 10
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(log10f(x_f2.x), log10f(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(log10f(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(log10f(x0), log10f(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(log10f(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return log10f(x);
            } else {
                return std::log10(x);
            }
        } else if constexpr (Mode == UnaryMode::Log1p) {
            // log1p: log(1 + x), numerically stable for values close to zero
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(log1pf(x_f2.x), log1pf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(log1pf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(log1pf(x0), log1pf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(log1pf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return log1pf(x);
            } else if constexpr (std::is_same_v<T, double>) {
                return log1p(x);
            } else {
                return std::log1p(x);
            }
        } else if constexpr (Mode == UnaryMode::Reciprocal) {
            if constexpr (std::is_same_v<T, half2>) {
                return h2rcp(x);
            } else if constexpr (std::is_same_v<T, half>) {
                return hrcp(x);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(__frcp_rn(x0), __frcp_rn(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(__frcp_rn(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return __frcp_rn(x);
            } else {
                return T(1) / x;
            }
        } else if constexpr (Mode == UnaryMode::Sqrt) {
            if constexpr (std::is_same_v<T, half2>) {
                return h2sqrt(x);
            } else if constexpr (std::is_same_v<T, half>) {
                return hsqrt(x);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(sqrtf(x0), sqrtf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(sqrtf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return __fsqrt_rn(x);
            } else {
                return std::sqrt(x);
            }
        } else if constexpr (Mode == UnaryMode::Square) {
            return x * x;
        } else if constexpr (Mode == UnaryMode::Rsqrt) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(__frsqrt_rn(x_f2.x), __frsqrt_rn(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(__frsqrt_rn(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(__frsqrt_rn(x0), __frsqrt_rn(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(__frsqrt_rn(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return __frsqrt_rn(x);
            } else {
                return T(1) / std::sqrt(x);
            }
        } else if constexpr (Mode == UnaryMode::Neg) {
            if constexpr (std::is_same_v<T, half2>) {
                return __hneg2(x);
            } else if constexpr (std::is_same_v<T, half>) {
                return __hneg(x);
            } else {
                return -x;
            }
        } else if constexpr (Mode == UnaryMode::Ceil) {
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
        } else if constexpr (Mode == UnaryMode::Floor) {
            if constexpr (std::is_same_v<T, half2>) {
                return h2floor(x);
            } else if constexpr (std::is_same_v<T, half>) {
                return hfloor(x);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(floorf(x0), floorf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(floorf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return floorf(x);
            } else if constexpr (std::is_integral_v<T>) {
                return x;
            } else {
                return std::floor(x);
            }
        } else if constexpr (Mode == UnaryMode::Round) {
            if constexpr (std::is_same_v<T, half2>) {
                return h2rint(x);
            } else if constexpr (std::is_same_v<T, half>) {
                return hrint(x);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(rintf(x0), rintf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(rintf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return rintf(x);
            } else if constexpr (std::is_integral_v<T>) {
                return x;
            } else {
                return std::nearbyint(x);
            }
        } else if constexpr (Mode == UnaryMode::Sin) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(__sinf(x_f2.x), __sinf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(__sinf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(sinf(x0), sinf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(sinf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return __sinf(x);
            } else {
                return std::sin(x);
            }
        } else if constexpr (Mode == UnaryMode::Cos) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(__cosf(x_f2.x), __cosf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(__cosf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(cosf(x0), cosf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(cosf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return __cosf(x);
            } else {
                return std::cos(x);
            }
        } else if constexpr (Mode == UnaryMode::Tan) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(tanf(x_f2.x), tanf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(tanf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return tanf(x);
            } else {
                return std::tan(x);
            }
        } else if constexpr (Mode == UnaryMode::Asin) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(asinf(x_f2.x), asinf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(asinf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return asinf(x);
            } else {
                return std::asin(x);
            }
        } else if constexpr (Mode == UnaryMode::Acos) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(acosf(x_f2.x), acosf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(acosf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return acosf(x);
            } else {
                return std::acos(x);
            }
        } else if constexpr (Mode == UnaryMode::Atan) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(atanf(x_f2.x), atanf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(atanf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return atanf(x);
            } else {
                return std::atan(x);
            }
        } else if constexpr (Mode == UnaryMode::Sinh) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(sinhf(x_f2.x), sinhf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(sinhf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return sinhf(x);
            } else {
                return std::sinh(x);
            }
        } else if constexpr (Mode == UnaryMode::Cosh) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(coshf(x_f2.x), coshf(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(coshf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return coshf(x);
            } else {
                return std::cosh(x);
            }
        } else if constexpr (Mode == UnaryMode::Tanh) {
            if constexpr (std::is_same_v<T, half2>) {
                return __h2tanh(x);
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(tanhf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float f0 = __bfloat162float(__low2bfloat16(x));
                float f1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(tanhf(f0), tanhf(f1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(tanhf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return tanhf(x);
            } else if constexpr (std::is_same_v<T, double>) {
                return ::tanh(x);
            } else {
                return std::tanh(x);
            }
        } else if constexpr (Mode == UnaryMode::Asinh) {
            if constexpr (std::is_same_v<T, half2>) {
                return __floats2half2_rn(asinhf(__half2float(__low2half(x))), asinhf(__half2float(__high2half(x))));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(asinhf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(asinhf(x0), asinhf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(asinhf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return asinhf(x);
            } else {
                return std::asinh(x);
            }
        } else if constexpr (Mode == UnaryMode::Acosh) {
            if constexpr (std::is_same_v<T, half2>) {
                return __floats2half2_rn(acoshf(__half2float(__low2half(x))), acoshf(__half2float(__high2half(x))));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(acoshf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(acoshf(x0), acoshf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(acoshf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return acoshf(x);
            } else {
                return std::acosh(x);
            }
        } else if constexpr (Mode == UnaryMode::Atanh) {
            if constexpr (std::is_same_v<T, half2>) {
                return __floats2half2_rn(atanhf(__half2float(__low2half(x))), atanhf(__half2float(__high2half(x))));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(atanhf(__half2float(x)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(atanhf(x0), atanhf(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                return __float2bfloat16_rn(atanhf(__bfloat162float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return atanhf(x);
            } else {
                return std::atanh(x);
            }
        } else if constexpr (Mode == UnaryMode::Relu) {
            if constexpr (std::is_same_v<T, half2>) {
                return __hmax2(x, __floats2half2_rn(0.0f, 0.0f));
            } else {
                return x > T(0) ? x : T(0);
            }
        } else if constexpr (Mode == UnaryMode::Sigmoid) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                float2 exp_neg_x = make_float2(__expf(-x_f2.x), __expf(-x_f2.y));
                return __float22half2_rn(make_float2(1.0f / (1.0f + exp_neg_x.x), 1.0f / (1.0f + exp_neg_x.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float x_ = __half2float(x);
                return __float2half(1.0f / (1.0f + __expf(-x_)));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 x_f2 = __bfloat1622float2(x);
                float2 exp_neg_x = make_float2(__expf(-x_f2.x), __expf(-x_f2.y));
                return __floats2bfloat162_rn(1.0f / (1.0f + exp_neg_x.x), 1.0f / (1.0f + exp_neg_x.y));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float x_ = __bfloat162float(x);
                return __float2bfloat16_rn(1.0f / (1.0f + __expf(-x_)));
            } else if constexpr (std::is_same_v<T, float>) {
                return 1.0f / (1.0f + __expf(-x));
            } else if constexpr (std::is_same_v<T, double>) {
                return 1.0 / (1.0 + exp(-x));
            } else {
                return T(1) / (T(1) + std::exp(-x));
            }
        } else if constexpr (Mode == UnaryMode::Sign) {
            if constexpr (std::is_same_v<T, half2>) {
                const auto lt_mask = __hlt2(x, __floats2half2_rn(0.0f, 0.0f));
                return __hadd2(__hneg2(lt_mask), __hsub2(__floats2half2_rn(1.0f, 1.0f), lt_mask));
            } else if constexpr (std::is_same_v<T, half>) {
                return x > half(0) ? half(1) : (x == half(0) ? half(0) : half(-1));
            } else {
                return x > T(0) ? T(1) : (x == T(0) ? T(0) : T(-1));
            }
        } else if constexpr (Mode == UnaryMode::Erf) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(erff(x_f2.x), erff(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                return __float2half(erff(__half2float(x)));
            } else if constexpr (std::is_same_v<T, float>) {
                return erff(x);
            } else {
                return std::erf(x);
            }
        } else if constexpr (Mode == UnaryMode::IsNan) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(
                    __isnanf(x_f2.x) ? 1.0f : 0.0f,
                    __isnanf(x_f2.y) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half>) {
                float x_ = __half2float(x);
                return __float2half(__isnanf(x_) ? 1.0f : 0.0f);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(
                    __isnanf(x0) ? 1.0f : 0.0f,
                    __isnanf(x1) ? 1.0f : 0.0f);
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float x_ = __bfloat162float(x);
                return __float2bfloat16_rn(__isnanf(x_) ? 1.0f : 0.0f);
            } else if constexpr (std::is_same_v<T, float>) {
                return __isnanf(x) ? 1.0f : 0.0f;
            } else if constexpr (std::is_same_v<T, double>) {
                return __isnan(x) ? 1.0 : 0.0;
            } else if constexpr (std::is_floating_point_v<T>) {
                return std::isnan(x) ? T(1) : T(0);
            } else {
                // For integral types, NaN doesn't exist, so always return 0
                return T(0);
            }
        } else if constexpr (Mode == UnaryMode::IsInf) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                return __float22half2_rn(make_float2(
                    __isinff(x_f2.x) ? 1.0f : 0.0f,
                    __isinff(x_f2.y) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half>) {
                float x_ = __half2float(x);
                return __float2half(__isinff(x_) ? 1.0f : 0.0f);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(
                    __isinff(x0) ? 1.0f : 0.0f,
                    __isinff(x1) ? 1.0f : 0.0f);
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float x_ = __bfloat162float(x);
                return __float2bfloat16_rn(__isinff(x_) ? 1.0f : 0.0f);
            } else if constexpr (std::is_same_v<T, float>) {
                return __isinff(x) ? 1.0f : 0.0f;
            } else if constexpr (std::is_same_v<T, double>) {
                return __isinf(x) ? 1.0 : 0.0;
            } else if constexpr (std::is_floating_point_v<T>) {
                return std::isinf(x) ? T(1) : T(0);
            } else {
                // For integral types, Inf doesn't exist, so always return 0
                return T(0);
            }
        } else if constexpr (Mode == UnaryMode::IsFinite) {
            // isfinite(x) = !isnan(x) && !isinf(x)
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                auto isfinite_f32 = [](float val) -> float {
                    return (!__isnanf(val) && !__isinff(val)) ? 1.0f : 0.0f;
                };
                return __float22half2_rn(make_float2(
                    isfinite_f32(x_f2.x),
                    isfinite_f32(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float x_ = __half2float(x);
                return __float2half((!__isnanf(x_) && !__isinff(x_)) ? 1.0f : 0.0f);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                auto isfinite_f32 = [](float val) -> float {
                    return (!__isnanf(val) && !__isinff(val)) ? 1.0f : 0.0f;
                };
                return __floats2bfloat162_rn(
                    isfinite_f32(x0),
                    isfinite_f32(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float x_ = __bfloat162float(x);
                return __float2bfloat16_rn((!__isnanf(x_) && !__isinff(x_)) ? 1.0f : 0.0f);
            } else if constexpr (std::is_same_v<T, float>) {
                return (!__isnanf(x) && !__isinff(x)) ? 1.0f : 0.0f;
            } else if constexpr (std::is_same_v<T, double>) {
                return (!__isnan(x) && !__isinf(x)) ? 1.0 : 0.0;
            } else if constexpr (std::is_floating_point_v<T>) {
                return std::isfinite(x) ? T(1) : T(0);
            } else {
                // For integral types, all values are finite, so always return 1
                return T(1);
            }
        } else if constexpr (Mode == UnaryMode::Sinc) {
            // sinc(x) = sin(x) / x, sinc(0) = 1
            // For small values, use Taylor expansion for numerical stability
            // sinc(x) ≈ 1 - x²/6 + x⁴/120 - x⁶/5040
            if constexpr (std::is_same_v<T, half2>) {
                float2 x_f2 = __half22float2(x);
                auto sinc_f32 = [](float val) -> float {
                    float abs_val = fabsf(val);
                    if (abs_val < 1e-2f) {
                        // Use Taylor expansion for small values: 1 - x²/6 + x⁴/120 - x⁶/5040
                        float x2 = val * val;
                        return 1.0f - x2 * (1.0f / 6.0f - x2 * (1.0f / 120.0f - x2 * (1.0f / 5040.0f)));
                    } else {
                        return __sinf(val) / val;
                    }
                };
                return __float22half2_rn(make_float2(
                    sinc_f32(x_f2.x),
                    sinc_f32(x_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float x_ = __half2float(x);
                float abs_x = fabsf(x_);
                if (abs_x < 1e-2f) {
                    float x2 = x_ * x_;
                    return __float2half(1.0f - x2 * (1.0f / 6.0f - x2 * (1.0f / 120.0f - x2 * (1.0f / 5040.0f))));
                } else {
                    return __float2half(__sinf(x_) / x_);
                }
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float x0 = __bfloat162float(__low2bfloat16(x));
                float x1 = __bfloat162float(__high2bfloat16(x));
                auto sinc_f32 = [](float val) -> float {
                    float abs_val = fabsf(val);
                    if (abs_val < 1e-2f) {
                        float x2 = val * val;
                        return 1.0f - x2 * (1.0f / 6.0f - x2 * (1.0f / 120.0f - x2 * (1.0f / 5040.0f)));
                    } else {
                        return sinf(val) / val;
                    }
                };
                return __floats2bfloat162_rn(
                    sinc_f32(x0),
                    sinc_f32(x1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float x_ = __bfloat162float(x);
                float abs_x = fabsf(x_);
                if (abs_x < 1e-2f) {
                    float x2 = x_ * x_;
                    return __float2bfloat16_rn(1.0f - x2 * (1.0f / 6.0f - x2 * (1.0f / 120.0f - x2 * (1.0f / 5040.0f))));
                } else {
                    return __float2bfloat16_rn(sinf(x_) / x_);
                }
            } else if constexpr (std::is_same_v<T, float>) {
                float abs_x = fabsf(x);
                if (abs_x < 1e-2f) {
                    float x2 = x * x;
                    return 1.0f - x2 * (1.0f / 6.0f - x2 * (1.0f / 120.0f - x2 * (1.0f / 5040.0f)));
                } else {
                    return __sinf(x) / x;
                }
            } else if constexpr (std::is_same_v<T, double>) {
                double abs_x = std::fabs(x);
                if (abs_x < 1e-6) {
                    double x2 = x * x;
                    return 1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 5040.0)));
                } else {
                    return std::sin(x) / x;
                }
            } else if constexpr (std::is_floating_point_v<T>) {
                T abs_x = std::abs(x);
                if (abs_x < T(1e-2)) {
                    T x2 = x * x;
                    return T(1) - x2 * (T(1) / T(6) - x2 * (T(1) / T(120) - x2 * (T(1) / T(5040))));
                } else {
                    return std::sin(x) / x;
                }
            } else {
                // For integral types, sinc is not well-defined, return 1 for 0, 0 otherwise
                return x == T(0) ? T(1) : T(0);
            }
        } else if constexpr (Mode == UnaryMode::Hardswish) {
            // Hardswish: f(x) = x * clamp(x + 3, 0, 6) / 6
            auto hswish_f32 = [](float x) -> float {
                float y = x + 3.0f;
                y = y < 0.0f ? 0.0f : (y > 6.0f ? 6.0f : y);
                return x * (y * (1.0f / 6.0f));
            };
            if constexpr (std::is_same_v<T, half2>) {
                float2 vf = __half22float2(x);
                float2 vr = make_float2(
                    hswish_f32(vf.x),
                    hswish_f32(vf.y));
                return __float22half2_rn(vr);
            } else if constexpr (std::is_same_v<T, half>) {
                float xf = __half2float(x);
                float yf = hswish_f32(xf);
                return __float2half_rn(yf);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float f0 = __bfloat162float(__low2bfloat16(x));
                float f1 = __bfloat162float(__high2bfloat16(x));
                return __floats2bfloat162_rn(hswish_f32(f0), hswish_f32(f1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float xf = __bfloat162float(x);
                return __float2bfloat16_rz(hswish_f32(xf));
            } else if constexpr (std::is_same_v<T, float>) {
                return hswish_f32(x);
            } else if constexpr (std::is_same_v<T, double>) {
                double xd = static_cast<double>(x);
                double yd = xd * (std::fmin(std::fmax(xd + 3.0, 0.0), 6.0) / 6.0);
                return static_cast<T>(yd);
            } else {
                double xd = static_cast<double>(x);
                double yd = xd * (std::fmin(std::fmax(xd + 3.0, 0.0), 6.0) / 6.0);
                return static_cast<T>(yd);
            }
        } else {
            static_assert(always_false<Mode>::value, "Unsupported unary operation mode");
            return x;
        }
    }
};
} // namespace cuda
#endif // __CUDACC__

/**
 * @brief Macro to define a unary elementwise descriptor for a specific operation.
 *
 * This macro simplifies the definition of unary operators (abs, log, sin, cos, etc.)
 * by automatically generating the Descriptor class and operation struct using the
 * ELEMENTWISE_DESCRIPTOR macro and UnaryOp template.
 *
 * Usage:
 *   UNARY_ELEMENTWISE_DESCRIPTOR(abs, cpu, UnaryMode::Abs)
 *   UNARY_ELEMENTWISE_DESCRIPTOR(log, cpu, UnaryMode::Log)
 *
 * @param OP        The operator name (e.g., abs, log, sin)
 * @param NAMESPACE The device namespace (e.g., cpu, nvidia)
 * @param MODE      The UnaryMode enum value for this operation
 */
#define UNARY_ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE, MODE) \
                                                          \
    ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE)                 \
                                                          \
    namespace op::OP::NAMESPACE {                         \
    using Op = op::elementwise::unary::UnaryOp<MODE>;     \
    }

} // namespace op::elementwise::unary

#endif // __INFINIOP_ELEMENTWISE_UNARY_H__
