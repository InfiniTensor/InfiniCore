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
    Pow,
    Mod,
    Max,
    Min,
    // Logical operations (for future use):
    // And, Or, Xor, Less, LessOrEqual, Equal, Greater, GreaterOrEqual
};

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
        } else if constexpr (Mode == BinaryMode::Pow) {
            return std::pow(a, b);
        } else if constexpr (Mode == BinaryMode::Mod) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmod(a, b);
            } else {
                return a % b;
            }
        } else if constexpr (Mode == BinaryMode::Max) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmax(a, b);
            } else {
                return std::max(a, b);
            }
        } else if constexpr (Mode == BinaryMode::Min) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmin(a, b);
            } else {
                return std::min(a, b);
            }
        } else {
            static_assert(Mode != Mode, "Unsupported binary operation mode");
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
        } else if constexpr (Mode == BinaryMode::Pow) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(__powf(a_f2.x, b_f2.x), __powf(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                float ans_f = __powf(a_, b_);
                return __float2half(isnan(ans_f) ? std::pow(a_, b_) : ans_f);
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
        } else if constexpr (Mode == BinaryMode::Mod) {
            if constexpr (std::is_same_v<T, half2>) {
                float2 a_f2 = __half22float2(a);
                float2 b_f2 = __half22float2(b);
                return __float22half2_rn(make_float2(std::fmod(a_f2.x, b_f2.x), std::fmod(a_f2.y, b_f2.y)));
            } else if constexpr (std::is_same_v<T, half>) {
                float a_ = __half2float(a);
                float b_ = __half2float(b);
                return __float2half(std::fmod(a_, b_));
            } else if constexpr (std::is_floating_point_v<T>) {
                return std::fmod(a, b);
            } else {
                return a % b;
            }
        } else if constexpr (Mode == BinaryMode::Max) {
            if constexpr (std::is_same_v<T, half2>) {
                return __hmax2(a, b);
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return a > b ? a : b;
            } else if constexpr (std::is_same_v<T, float>) {
                return fmaxf(a, b);
            } else {
                return a > b ? a : b;
            }
        } else if constexpr (Mode == BinaryMode::Min) {
            if constexpr (std::is_same_v<T, half2>) {
                return __hmin2(a, b);
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                return a < b ? a : b;
            } else if constexpr (std::is_same_v<T, float>) {
                return fminf(a, b);
            } else {
                return a < b ? a : b;
            }
        } else {
            static_assert(Mode != Mode, "Unsupported binary operation mode");
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
