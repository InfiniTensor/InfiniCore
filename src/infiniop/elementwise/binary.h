#ifndef __INFINIOP_ELEMENTWISE_BINARY_H__
#define __INFINIOP_ELEMENTWISE_BINARY_H__

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

namespace op::elementwise::binary {

/**
 * @brief Represents all the currently defined binary operations.
 *
 * This enum is used to specify which binary operation to perform
 * in the generic BinaryOp template.
 */
enum class BinaryMode {
    // Arithmetic operations:
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide, // floor_divide: floor(a / b)
    Pow,
    CopySign,
    Hypot,
    Atan2, // atan2: atan2(y, x)
    Mod,
    Remainder,
    Max,
    Min,
    Fmax,
    Fmin,
    // Comparison operations:
    Greater,        // gt: a > b
    Less,           // lt: a < b
    GreaterOrEqual, // ge: a >= b
    LessOrEqual,    // le: a <= b
    Equal,          // eq: a == b
    NotEqual,       // ne: a != b
    // Logical operations:
    LogicalAnd, // logical_and: a && b (non-zero as true)
    LogicalOr,  // logical_or: a || b (non-zero as true)
    LogicalXor, // logical_xor: a ^ b (exactly one non-zero as true)
    // Bitwise operations:
    BitwiseAnd,        // bitwise_and: a & b (only for integral types)
    BitwiseOr,         // bitwise_or: a | b (only for integral types)
    BitwiseXor,        // bitwise_xor: a ^ b (only for integral types)
    BitwiseLeftShift,  // bitwise_left_shift: a << b (only for integral types)
    BitwiseRightShift, // bitwise_right_shift: a >> b (only for integral types)
};

// Helper template for static_assert in else branches
template <BinaryMode M>
struct always_false : std::false_type {};

/**
 * @brief Generic binary operation template that performs different operations
 *        based on the specified BinaryMode.
 *
 * This template allows multiple binary operators (pow, div, mod, min, max, etc.)
 * to share the same implementation infrastructure while only differing in the
 * operation mode.
 *
 * @tparam Mode The binary operation mode (from BinaryMode enum)
 */
template <BinaryMode Mode>
struct BinaryOp {
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (Mode == BinaryMode::Add) {
            return a + b;
        } else if constexpr (Mode == BinaryMode::Subtract) {
            return a - b;
        } else if constexpr (Mode == BinaryMode::Multiply) {
            return a * b;
        } else if constexpr (Mode == BinaryMode::Divide) {
            return a / b;
        } else if constexpr (Mode == BinaryMode::FloorDivide) {
            // Floor divide: floor(a / b)
            if constexpr (std::is_integral_v<T>) {
                // For integral types, integer division is already floor division
                return a / b;
            } else {
                // For floating point types, use std::floor
                return std::floor(a / b);
            }
        } else if constexpr (Mode == BinaryMode::Pow) {
            return std::pow(a, b);
        } else if constexpr (Mode == BinaryMode::CopySign) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::copysign(a, b);
            } else {
                // For integral types, return a with sign of b
                return (b < T(0)) ? -std::abs(a) : std::abs(a);
            }
        } else if constexpr (Mode == BinaryMode::Hypot) {
            return std::hypot(a, b);
        } else if constexpr (Mode == BinaryMode::Atan2) {
            // atan2(y, x): returns the angle whose tangent is y/x
            return std::atan2(a, b);
        } else if constexpr (Mode == BinaryMode::Mod) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmod(a, b);
            } else {
                return a % b;
            }
        } else if constexpr (Mode == BinaryMode::Remainder) {
            if constexpr (std::is_floating_point_v<T>) {
                // PyTorch remainder: x - floor(x/y) * y, result sign matches divisor (y)
                T quotient = std::floor(a / b);
                return a - quotient * b;
            } else {
                // For integral types, remainder is same as mod
                return a % b;
            }
        } else if constexpr (Mode == BinaryMode::Max) {
            // Max: propagates NaN (if either is NaN, result is NaN)
            if constexpr (std::is_floating_point_v<T>) {
                // Use std::max which propagates NaN (a > b ? a : b behavior with NaN)
                return (a > b) ? a : b;
            } else {
                return std::max(a, b);
            }
        } else if constexpr (Mode == BinaryMode::Min) {
            // Min: propagates NaN (if either is NaN, result is NaN)
            if constexpr (std::is_floating_point_v<T>) {
                // Use std::min which propagates NaN (a < b ? a : b behavior with NaN)
                return (a < b) ? a : b;
            } else {
                return std::min(a, b);
            }
        } else if constexpr (Mode == BinaryMode::Fmax) {
            // Fmax: ignores NaN (if one is NaN, return the other)
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmax(a, b);
            } else {
                return std::max(a, b);
            }
        } else if constexpr (Mode == BinaryMode::Fmin) {
            // Fmin: ignores NaN (if one is NaN, return the other)
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmin(a, b);
            } else {
                return std::min(a, b);
            }
        } else if constexpr (Mode == BinaryMode::Greater) {
            // Return 1.0 if a > b, else 0.0
            return static_cast<T>(a > b ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::Less) {
            // Return 1.0 if a < b, else 0.0
            return static_cast<T>(a < b ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::GreaterOrEqual) {
            // Return 1.0 if a >= b, else 0.0
            return static_cast<T>(a >= b ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::LessOrEqual) {
            // Return 1.0 if a <= b, else 0.0
            return static_cast<T>(a <= b ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::Equal) {
            // Return 1.0 if a == b, else 0.0
            return static_cast<T>(a == b ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::NotEqual) {
            // Return 1.0 if a != b, else 0.0
            return static_cast<T>(a != b ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::LogicalAnd) {
            // Return 1.0 if both a and b are non-zero, else 0.0
            return static_cast<T>((a != T(0) && b != T(0)) ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::LogicalOr) {
            // Return 1.0 if either a or b is non-zero, else 0.0
            return static_cast<T>((a != T(0) || b != T(0)) ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::LogicalXor) {
            // Return 1.0 if exactly one of a or b is non-zero, else 0.0
            bool a_nonzero = (a != T(0));
            bool b_nonzero = (b != T(0));
            return static_cast<T>((a_nonzero != b_nonzero) ? T(1) : T(0));
        } else if constexpr (Mode == BinaryMode::BitwiseAnd) {
            // Bitwise AND: a & b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a & b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseOr) {
            // Bitwise OR: a | b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a | b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseXor) {
            // Bitwise XOR: a ^ b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a ^ b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseLeftShift) {
            // Bitwise left shift: a << b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a << b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseRightShift) {
            // Bitwise right shift: a >> b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a >> b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else {
            static_assert(always_false<Mode>::value, "Unsupported binary operation mode");
            return a;
        }
    }
};

#ifdef __CUDACC__
/**
 * @brief CUDA-specific binary operation template that performs different operations
 *        based on the specified BinaryMode, using CUDA-optimized functions.
 *
 * This template provides CUDA device functions optimized for GPU execution,
 * using intrinsics like __powf, __h2div, __hmin2, __hmax2, etc.
 *
 * @tparam Mode The binary operation mode (from BinaryMode enum)
 */
namespace cuda {
template <BinaryMode Mode>
struct BinaryOp {
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (Mode == BinaryMode::Add) {
            if constexpr (std::is_same_v<T, half2>) {
                return __hadd2(a, b);
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return __hadd(a, b);
            } else if constexpr (std::is_same_v<T, float>) {
                return __fadd_rn(a, b);
            } else {
                return a + b;
            }
        } else if constexpr (Mode == BinaryMode::Subtract) {
            if constexpr (std::is_same_v<T, half2>) {
                return __hsub2(a, b);
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return __hsub(a, b);
            } else if constexpr (std::is_same_v<T, float>) {
                return __fsub_rn(a, b);
            } else {
                return a - b;
            }
        } else if constexpr (Mode == BinaryMode::Multiply) {
            if constexpr (std::is_same_v<T, half2>) {
                return __hmul2(a, b);
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return __hmul(a, b);
            } else if constexpr (std::is_same_v<T, float>) {
                return __fmul_rd(a, b);
            } else {
                return a * b;
            }
        } else if constexpr (Mode == BinaryMode::Divide) {
            if constexpr (std::is_same_v<T, half2>) {
                return __h2div(a, b);
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return a / b;
            } else if constexpr (std::is_same_v<T, float>) {
                return __fdividef(a, b);
            } else {
                return a / b;
            }
        } else if constexpr (Mode == BinaryMode::FloorDivide) {
            // Floor divide: floor(a / b)
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(floorf(a_f2.x / b_f2.x), floorf(a_f2.y / b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                return __float2half(floorf(a_ / b_));
            } else if constexpr (std::is_integral_v<T>) {
                // For integral types, integer division is already floor division
                return a / b;
            } else if constexpr (std::is_same_v<T, float>) {
                return floorf(a / b);
            } else {
                return std::floor(a / b);
            }
        } else if constexpr (Mode == BinaryMode::Pow) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(__powf(a_f2.x, b_f2.x), __powf(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                // Use __powf only (std::pow is host function, cannot be used in device code)
                return __float2half(__powf(a_, b_));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 a_f2 = __bfloat1622float2(a);
                float2 b_f2 = __bfloat1622float2(b);
                return __floats2bfloat162_rn(__powf(a_f2.x, b_f2.x), __powf(a_f2.y, b_f2.y));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float a_ = __bfloat162float(a);
                float b_ = __bfloat162float(b);
                return __float2bfloat16_rn(__powf(a_, b_));
            } else if constexpr (std::is_same_v<T, float>) {
                return __powf(a, b);
            } else {
                return std::pow(a, b);
            }
        } else if constexpr (Mode == BinaryMode::CopySign) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(copysignf(a_f2.x, b_f2.x), copysignf(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                return __float2half(copysignf(a_, b_));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 a_f2 = __bfloat1622float2(a);
                float2 b_f2 = __bfloat1622float2(b);
                return __floats2bfloat162_rn(copysignf(a_f2.x, b_f2.x), copysignf(a_f2.y, b_f2.y));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float a_ = __bfloat162float(a);
                float b_ = __bfloat162float(b);
                return __float2bfloat16_rn(copysignf(a_, b_));
            } else if constexpr (std::is_same_v<T, float>) {
                return copysignf(a, b);
            } else if constexpr (std::is_floating_point_v<T>) {
                return std::copysign(a, b);
            } else {
                // For integral types, return a with sign of b
                return (b < T(0)) ? -std::abs(a) : std::abs(a);
            }
        } else if constexpr (Mode == BinaryMode::Hypot) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(hypotf(a_f2.x, b_f2.x), hypotf(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                return __float2half(hypotf(a_, b_));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 a_f2 = __bfloat1622float2(a);
                float2 b_f2 = __bfloat1622float2(b);
                return __floats2bfloat162_rn(hypotf(a_f2.x, b_f2.x), hypotf(a_f2.y, b_f2.y));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float a_ = __bfloat162float(a);
                float b_ = __bfloat162float(b);
                return __float2bfloat16_rn(hypotf(a_, b_));
            } else if constexpr (std::is_same_v<T, float>) {
                return hypotf(a, b);
            } else {
                return std::hypot(a, b);
            }
        } else if constexpr (Mode == BinaryMode::Atan2) {
            // atan2(y, x): returns the angle whose tangent is y/x
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(atan2f(a_f2.x, b_f2.x), atan2f(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                return __float2half(atan2f(a_, b_));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 a_f2 = __bfloat1622float2(a);
                float2 b_f2 = __bfloat1622float2(b);
                return __floats2bfloat162_rn(atan2f(a_f2.x, b_f2.x), atan2f(a_f2.y, b_f2.y));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float a_ = __bfloat162float(a);
                float b_ = __bfloat162float(b);
                return __float2bfloat16_rn(atan2f(a_, b_));
            } else if constexpr (std::is_same_v<T, float>) {
                return atan2f(a, b);
            } else {
                return std::atan2(a, b);
            }
        } else if constexpr (Mode == BinaryMode::Mod) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(fmodf(a_f2.x, b_f2.x), fmodf(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                return __float2half(fmodf(a_, b_));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 a_f2 = __bfloat1622float2(a);
                float2 b_f2 = __bfloat1622float2(b);
                return __floats2bfloat162_rn(fmodf(a_f2.x, b_f2.x), fmodf(a_f2.y, b_f2.y));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float a_ = __bfloat162float(a);
                float b_ = __bfloat162float(b);
                return __float2bfloat16_rn(fmodf(a_, b_));
            } else if constexpr (std::is_floating_point_v<T>) {
                return fmodf(a, b);
            } else {
                return a % b;
            }
        } else if constexpr (Mode == BinaryMode::Remainder) {
            // PyTorch remainder: x - floor(x/y) * y, result sign matches divisor (y)
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                float2 q_f2 = make_float2(floorf(a_f2.x / b_f2.x), floorf(a_f2.y / b_f2.y));
                float2 r_f2 = make_float2(a_f2.x - q_f2.x * b_f2.x, a_f2.y - q_f2.y * b_f2.y);
                return __float22half2_rn(r_f2);
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                float q_ = floorf(a_ / b_);
                float r_ = a_ - q_ * b_;
                return __float2half(r_);
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float2 a_f2 = __bfloat1622float2(a);
                float2 b_f2 = __bfloat1622float2(b);
                float2 q_f2 = make_float2(floorf(a_f2.x / b_f2.x), floorf(a_f2.y / b_f2.y));
                float2 r_f2 = make_float2(a_f2.x - q_f2.x * b_f2.x, a_f2.y - q_f2.y * b_f2.y);
                return __floats2bfloat162_rn(r_f2.x, r_f2.y);
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float a_ = __bfloat162float(a);
                float b_ = __bfloat162float(b);
                float q_ = floorf(a_ / b_);
                float r_ = a_ - q_ * b_;
                return __float2bfloat16_rn(r_);
            } else if constexpr (std::is_same_v<T, float>) {
                float q = floorf(a / b);
                return a - q * b;
            } else if constexpr (std::is_floating_point_v<T>) {
                T quotient = std::floor(a / b);
                return a - quotient * b;
            } else {
                // For integral types, remainder is same as mod
                return a % b;
            }
        } else if constexpr (Mode == BinaryMode::Max) {
            // Max: propagates NaN (torch.maximum behavior)
            if constexpr (std::is_same_v<T, half2>) {
                return __hmax2(a, b);
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                // For half/bfloat16, use comparison which propagates NaN
                return a > b ? a : b;
            } else if constexpr (std::is_same_v<T, float>) {
                // For float, use comparison which propagates NaN
                return a > b ? a : b;
            } else {
                return a > b ? a : b;
            }
        } else if constexpr (Mode == BinaryMode::Min) {
            // Min: propagates NaN (torch.minimum behavior)
            if constexpr (std::is_same_v<T, half2>) {
                return __hmin2(a, b);
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                // For half/bfloat16, use comparison which propagates NaN
                return a < b ? a : b;
            } else if constexpr (std::is_same_v<T, float>) {
                // For float, use comparison which propagates NaN
                return a < b ? a : b;
            } else {
                return a < b ? a : b;
            }
        } else if constexpr (Mode == BinaryMode::Fmax) {
            // Fmax: ignores NaN (torch.fmax behavior - if one is NaN, return the other)
            if constexpr (std::is_same_v<T, half2>) {
                // __hmax2 may propagate NaN, so implement custom NaN-ignoring version
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(fmaxf(a_f2.x, b_f2.x), fmaxf(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                return __float2half(fmaxf(a_, b_));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float a0 = __bfloat162float(__low2bfloat16(a));
                float a1 = __bfloat162float(__high2bfloat16(a));
                float b0 = __bfloat162float(__low2bfloat16(b));
                float b1 = __bfloat162float(__high2bfloat16(b));
                return __floats2bfloat162_rn(fmaxf(a0, b0), fmaxf(a1, b1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float a_ = __bfloat162float(a);
                float b_ = __bfloat162float(b);
                return __float2bfloat16_rn(fmaxf(a_, b_));
            } else if constexpr (std::is_same_v<T, float>) {
                return fmaxf(a, b);
            } else if constexpr (std::is_same_v<T, double>) {
                return fmax(a, b);
            } else {
                return a > b ? a : b;
            }
        } else if constexpr (Mode == BinaryMode::Fmin) {
            // Fmin: ignores NaN (torch.fmin behavior - if one is NaN, return the other)
            if constexpr (std::is_same_v<T, half2>) {
                // __hmin2 may propagate NaN, so implement custom NaN-ignoring version
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(fminf(a_f2.x, b_f2.x), fminf(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                return __float2half(fminf(a_, b_));
            } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
                float a0 = __bfloat162float(__low2bfloat16(a));
                float a1 = __bfloat162float(__high2bfloat16(a));
                float b0 = __bfloat162float(__low2bfloat16(b));
                float b1 = __bfloat162float(__high2bfloat16(b));
                return __floats2bfloat162_rn(fminf(a0, b0), fminf(a1, b1));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float a_ = __bfloat162float(a);
                float b_ = __bfloat162float(b);
                return __float2bfloat16_rn(fminf(a_, b_));
            } else if constexpr (std::is_same_v<T, float>) {
                return fminf(a, b);
            } else if constexpr (std::is_same_v<T, double>) {
                return fmin(a, b);
            } else {
                return a < b ? a : b;
            }
        } else if constexpr (Mode == BinaryMode::Greater) {
            // Return 1.0 if a > b, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(
                    (a_f2.x > b_f2.x) ? 1.0f : 0.0f,
                    (a_f2.y > b_f2.y) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return (a > b) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                return (a > b) ? 1.0f : 0.0f;
            } else {
                return static_cast<T>((a > b) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::Less) {
            // Return 1.0 if a < b, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(
                    (a_f2.x < b_f2.x) ? 1.0f : 0.0f,
                    (a_f2.y < b_f2.y) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return (a < b) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                return (a < b) ? 1.0f : 0.0f;
            } else {
                return static_cast<T>((a < b) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::GreaterOrEqual) {
            // Return 1.0 if a >= b, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(
                    (a_f2.x >= b_f2.x) ? 1.0f : 0.0f,
                    (a_f2.y >= b_f2.y) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return (a >= b) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                return (a >= b) ? 1.0f : 0.0f;
            } else {
                return static_cast<T>((a >= b) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::LessOrEqual) {
            // Return 1.0 if a <= b, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(
                    (a_f2.x <= b_f2.x) ? 1.0f : 0.0f,
                    (a_f2.y <= b_f2.y) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return (a <= b) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                return (a <= b) ? 1.0f : 0.0f;
            } else {
                return static_cast<T>((a <= b) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::Equal) {
            // Return 1.0 if a == b, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(
                    (a_f2.x == b_f2.x) ? 1.0f : 0.0f,
                    (a_f2.y == b_f2.y) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return (a == b) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                return (a == b) ? 1.0f : 0.0f;
            } else {
                return static_cast<T>((a == b) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::NotEqual) {
            // Return 1.0 if a != b, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(
                    (a_f2.x != b_f2.x) ? 1.0f : 0.0f,
                    (a_f2.y != b_f2.y) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return (a != b) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                return (a != b) ? 1.0f : 0.0f;
            } else {
                return static_cast<T>((a != b) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::LogicalAnd) {
            // Return 1.0 if both a and b are non-zero, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(
                    ((a_f2.x != 0.0f) && (b_f2.x != 0.0f)) ? 1.0f : 0.0f,
                    ((a_f2.y != 0.0f) && (b_f2.y != 0.0f)) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return ((a != T(0)) && (b != T(0))) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                return ((a != 0.0f) && (b != 0.0f)) ? 1.0f : 0.0f;
            } else {
                return static_cast<T>(((a != T(0)) && (b != T(0))) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::LogicalOr) {
            // Return 1.0 if either a or b is non-zero, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(
                    ((a_f2.x != 0.0f) || (b_f2.x != 0.0f)) ? 1.0f : 0.0f,
                    ((a_f2.y != 0.0f) || (b_f2.y != 0.0f)) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return ((a != T(0)) || (b != T(0))) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                return ((a != 0.0f) || (b != 0.0f)) ? 1.0f : 0.0f;
            } else {
                return static_cast<T>(((a != T(0)) || (b != T(0))) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::LogicalXor) {
            // Return 1.0 if exactly one of a or b is non-zero, else 0.0
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                bool a_x_nonzero = (a_f2.x != 0.0f);
                bool b_x_nonzero = (b_f2.x != 0.0f);
                bool a_y_nonzero = (a_f2.y != 0.0f);
                bool b_y_nonzero = (b_f2.y != 0.0f);
                return __float22half2_rn(make_float2(
                    (a_x_nonzero != b_x_nonzero) ? 1.0f : 0.0f,
                    (a_y_nonzero != b_y_nonzero) ? 1.0f : 0.0f));
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                bool a_nonzero = (a != T(0));
                bool b_nonzero = (b != T(0));
                return (a_nonzero != b_nonzero) ? T(1) : T(0);
            } else if constexpr (std::is_same_v<T, float>) {
                bool a_nonzero = (a != 0.0f);
                bool b_nonzero = (b != 0.0f);
                return (a_nonzero != b_nonzero) ? 1.0f : 0.0f;
            } else {
                bool a_nonzero = (a != T(0));
                bool b_nonzero = (b != T(0));
                return static_cast<T>((a_nonzero != b_nonzero) ? 1 : 0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseAnd) {
            // Bitwise AND: a & b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a & b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseOr) {
            // Bitwise OR: a | b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a | b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseXor) {
            // Bitwise XOR: a ^ b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a ^ b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseLeftShift) {
            // Bitwise left shift: a << b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a << b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else if constexpr (Mode == BinaryMode::BitwiseRightShift) {
            // Bitwise right shift: a >> b (only for integral types)
            if constexpr (std::is_integral_v<T>) {
                return a >> b;
            } else {
                static_assert(std::is_integral_v<T>, "Bitwise operations require integral types");
                return T(0);
            }
        } else {
            static_assert(always_false<Mode>::value, "Unsupported binary operation mode");
            return a;
        }
    }
};
} // namespace cuda
#endif // __CUDACC__

/**
 * @brief Macro to define a binary elementwise descriptor for a specific operation.
 *
 * This macro simplifies the definition of binary operators (pow, div, mod, min, max, etc.)
 * by automatically generating the Descriptor class and operation struct using the
 * ELEMENTWISE_DESCRIPTOR macro and BinaryOp template.
 *
 * Usage:
 *   BINARY_ELEMENTWISE_DESCRIPTOR(pow, cpu, BinaryMode::Pow)
 *   BINARY_ELEMENTWISE_DESCRIPTOR(div, cpu, BinaryMode::Divide)
 *
 * @param OP        The operator name (e.g., pow, div, mod)
 * @param NAMESPACE The device namespace (e.g., cpu, nvidia)
 * @param MODE      The BinaryMode enum value for this operation
 */
#define BINARY_ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE, MODE) \
                                                           \
    ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE)                  \
                                                           \
    namespace op::OP::NAMESPACE {                          \
    using Op = op::elementwise::binary::BinaryOp<MODE>;    \
    }

/**
 * @brief Macro to define a binary elementwise descriptor for CUDA/NVIDIA backend.
 *
 * This macro is similar to BINARY_ELEMENTWISE_DESCRIPTOR but uses the CUDA-specific
 * BinaryOp implementation for better GPU performance.
 *
 * Usage:
 *   BINARY_ELEMENTWISE_DESCRIPTOR_CUDA(pow, nvidia, BinaryMode::Pow)
 *   BINARY_ELEMENTWISE_DESCRIPTOR_CUDA(div, nvidia, BinaryMode::Divide)
 *
 * @param OP        The operator name (e.g., pow, div, mod)
 * @param NAMESPACE The device namespace (e.g., nvidia)
 * @param MODE      The BinaryMode enum value for this operation
 */
#ifdef __CUDACC__
#define BINARY_ELEMENTWISE_DESCRIPTOR_CUDA(OP, NAMESPACE, MODE) \
                                                                \
    ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE)                       \
                                                                \
    namespace op::OP::cuda {                                    \
    using Op = op::elementwise::binary::cuda::BinaryOp<MODE>;   \
    }
#endif // __CUDACC__

} // namespace op::elementwise::binary

#endif // __INFINIOP_ELEMENTWISE_BINARY_H__
