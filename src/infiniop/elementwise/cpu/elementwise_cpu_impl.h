#ifndef __INFINIOP_ELEMENTWISE_CPU_IMPL_H__
#define __INFINIOP_ELEMENTWISE_CPU_IMPL_H__

#include "../../../utils/check.h"
#include "../../../utils/result.hpp"
#include "../../devices/cpu/common_cpu.h"
#include "elementwise_cpu.h"

/**
 * @brief Generic implementation for elementwise CPU operators.
 *
 * This file provides a generic implementation template that can be used
 * by all binary and unary operators to reduce code duplication.
 *
 * Usage:
 *   #include "elementwise_cpu_impl.h"
 *   namespace op::pow::cpu {
 *       using Op = op::elementwise::binary::BinaryOp<BinaryMode::Pow>;
 *       ELEMENTWISE_CPU_IMPL_BINARY(pow)
 *   }
 *
 *   namespace op::sqrt::cpu {
 *       using Op = op::elementwise::unary::UnaryOp<UnaryMode::Sqrt>;
 *       ELEMENTWISE_CPU_IMPL_UNARY(sqrt)
 *   }
 */

// =========================================================================
//  Internal Helpers (Private Macros to reduce duplication)
// =========================================================================

/**
 * @brief Common Calculate Switch Cases (F16 & F32)
 */
#define _IMPL_CALC_CASES_COMMON                                                             \
    case INFINI_DTYPE_F16:                                                                  \
        return _device_info->template calculate<Op, fp16_t>(_info, output, inputs, stream); \
    case INFINI_DTYPE_F32:                                                                  \
        return _device_info->template calculate<Op, float>(_info, output, inputs, stream);

/**
 * @brief Extended Calculate Switch Cases (Adds F64 & BF16)
 */
#define _IMPL_CALC_CASES_EXTENDED                                                           \
    _IMPL_CALC_CASES_COMMON                                                                 \
    case INFINI_DTYPE_F64:                                                                  \
        return _device_info->template calculate<Op, double>(_info, output, inputs, stream); \
    case INFINI_DTYPE_BF16:                                                                 \
        return _device_info->template calculate<Op, bf16_t>(_info, output, inputs, stream);

/**
 * @brief Integral Calculate Switch Cases (I32, I64, U8)
 * For bitwise operations that only support integral types
 */
#define _IMPL_CALC_CASES_INTEGRAL                                                            \
    case INFINI_DTYPE_I32:                                                                   \
        return _device_info->template calculate<Op, int32_t>(_info, output, inputs, stream); \
    case INFINI_DTYPE_I64:                                                                   \
        return _device_info->template calculate<Op, int64_t>(_info, output, inputs, stream); \
    case INFINI_DTYPE_U8:                                                                    \
        return _device_info->template calculate<Op, uint8_t>(_info, output, inputs, stream);

/**
 * @brief Generic Template for the Calculate method
 * @param CASES_MACRO The macro containing the switch cases to use
 */
#define _IMPL_CALCULATE_METHOD(CASES_MACRO)        \
    infiniStatus_t Descriptor::calculate(          \
        void *workspace,                           \
        size_t workspace_size,                     \
        void *output,                              \
        std::vector<const void *> inputs,          \
        void *stream) const {                      \
        switch (_dtype) {                          \
            CASES_MACRO                            \
        default:                                   \
            return INFINI_STATUS_BAD_TENSOR_DTYPE; \
        }                                          \
    }

/**
 * @brief Generic Template for the Create method
 * @param SHAPE_CHECK_BLOCK Code block to execute for shape checking
 * @param ... Variadic arguments for allowed data types in CHECK_DTYPE
 */
#define _IMPL_CREATE_METHOD(SHAPE_CHECK_BLOCK, ...)                                 \
    Descriptor::~Descriptor() = default;                                            \
    infiniStatus_t Descriptor::create(                                              \
        infiniopHandle_t handle_,                                                   \
        Descriptor **desc_ptr,                                                      \
        infiniopTensorDescriptor_t out_desc,                                        \
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) {                   \
        auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);             \
        auto dtype = out_desc->dtype();                                             \
        const auto &out_shape = out_desc->shape();                                  \
        SHAPE_CHECK_BLOCK                                                           \
        CHECK_DTYPE(dtype, __VA_ARGS__);                                            \
        CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec); \
        return INFINI_STATUS_SUCCESS;                                               \
    }

// =========================================================================
//  Public API Implementation Macros
// =========================================================================

/**
 * @brief Implementation for Binary Operators (F16, F32)
 *
 * This macro generates the Descriptor destructor, create, and calculate methods
 * for binary operators, using the generic implementation.
 *
 * Usage:
 *   namespace op::pow::cpu {
 *       using Op = op::elementwise::binary::BinaryOp<BinaryMode::Pow>;
 *       ELEMENTWISE_CPU_IMPL_BINARY(pow)
 *   }
 */
#define ELEMENTWISE_CPU_IMPL_BINARY(OP)                                                   \
    _IMPL_CREATE_METHOD(                                                                  \
        const auto &a_desc = input_desc_vec.at(0);                                        \
        const auto &b_desc = input_desc_vec.at(1);                                        \
        const auto &a_shape = a_desc->shape();                                            \
        const auto &b_shape = b_desc->shape();                                            \
        CHECK_SAME_SHAPE(out_shape, a_shape, b_shape);,                                   \
                                                      INFINI_DTYPE_F16, INFINI_DTYPE_F32) \
    _IMPL_CALCULATE_METHOD(_IMPL_CALC_CASES_COMMON)

/**
 * @brief Implementation for Unary Operators (F16, F32)
 *
 * This macro generates the Descriptor destructor, create, and calculate methods
 * for unary operators, using the generic implementation.
 *
 * Usage:
 *   namespace op::sqrt::cpu {
 *       using Op = op::elementwise::unary::UnaryOp<UnaryMode::Sqrt>;
 *       ELEMENTWISE_CPU_IMPL_UNARY(sqrt)
 *   }
 */
#define ELEMENTWISE_CPU_IMPL_UNARY(OP)                                           \
    _IMPL_CREATE_METHOD(                                                         \
        const auto &x_desc = input_desc_vec.at(0);                               \
        const auto &x_shape = x_desc->shape();                                   \
        CHECK_SAME_SHAPE(out_shape, x_shape);,                                   \
                                             INFINI_DTYPE_F16, INFINI_DTYPE_F32) \
    _IMPL_CALCULATE_METHOD(_IMPL_CALC_CASES_COMMON)

/**
 * @brief Implementation for Unary Operators Extended (F16, F32, F64, BF16)
 *
 * This macro generates the Descriptor destructor, create, and calculate methods
 * for unary operators supporting F16, F32, F64, and BF16 data types.
 *
 * Usage:
 *   namespace op::exp::cpu {
 *       using Op = op::elementwise::unary::UnaryOp<UnaryMode::Exp>;
 *       ELEMENTWISE_CPU_IMPL_UNARY_EXTENDED(exp)
 *   }
 */
#define ELEMENTWISE_CPU_IMPL_UNARY_EXTENDED(OP)                                                                       \
    _IMPL_CREATE_METHOD(                                                                                              \
        const auto &x_desc = input_desc_vec.at(0);                                                                    \
        const auto &x_shape = x_desc->shape();                                                                        \
        CHECK_SAME_SHAPE(out_shape, x_shape);,                                                                        \
                                             INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16) \
    _IMPL_CALCULATE_METHOD(_IMPL_CALC_CASES_EXTENDED)

/**
 * @brief Implementation for Binary Operators with Integral Types (I32, I64, U8)
 *
 * This macro generates the Descriptor destructor, create, and calculate methods
 * for binary operators that only support integral types (e.g., bitwise operations).
 *
 * Usage:
 *   namespace op::bitwise_and::cpu {
 *       using Op = op::elementwise::binary::BinaryOp<BinaryMode::BitwiseAnd>;
 *       ELEMENTWISE_CPU_IMPL_BINARY_INTEGRAL(bitwise_and)
 *   }
 */
#define ELEMENTWISE_CPU_IMPL_BINARY_INTEGRAL(OP)                                                           \
    _IMPL_CREATE_METHOD(                                                                                   \
        const auto &a_desc = input_desc_vec.at(0);                                                         \
        const auto &b_desc = input_desc_vec.at(1);                                                         \
        const auto &a_shape = a_desc->shape();                                                             \
        const auto &b_shape = b_desc->shape();                                                             \
        CHECK_SAME_SHAPE(out_shape, a_shape, b_shape);,                                                    \
                                                      INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U8) \
    _IMPL_CALCULATE_METHOD(_IMPL_CALC_CASES_INTEGRAL)

#endif // __INFINIOP_ELEMENTWISE_CPU_IMPL_H__
