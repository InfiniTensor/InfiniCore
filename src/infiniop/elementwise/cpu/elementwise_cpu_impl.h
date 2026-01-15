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

/**
 * @brief Macro to generate binary operator implementation.
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
#define ELEMENTWISE_CPU_IMPL_BINARY(OP)                                             \
                                                                                    \
    Descriptor::~Descriptor() = default;                                            \
                                                                                    \
    infiniStatus_t Descriptor::create(                                              \
        infiniopHandle_t handle_,                                                   \
        Descriptor **desc_ptr,                                                      \
        infiniopTensorDescriptor_t out_desc,                                        \
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) {                   \
        auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);             \
        auto dtype = out_desc->dtype();                                             \
        const auto &a_desc = input_desc_vec.at(0);                                  \
        const auto &b_desc = input_desc_vec.at(1);                                  \
        const auto &out_shape = out_desc->shape();                                  \
        const auto &a_shape = a_desc->shape();                                      \
        const auto &b_shape = b_desc->shape();                                      \
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);                     \
        CHECK_SAME_SHAPE(out_shape, a_shape, b_shape);                              \
        CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec); \
        return INFINI_STATUS_SUCCESS;                                               \
    }                                                                               \
                                                                                    \
    infiniStatus_t Descriptor::calculate(                                           \
        void *workspace,                                                            \
        size_t workspace_size,                                                      \
        void *output,                                                               \
        std::vector<const void *> inputs,                                           \
        void *stream) const {                                                       \
        switch (_dtype) {                                                           \
        case INFINI_DTYPE_F16:                                                      \
            return _device_info->template calculate<Op, fp16_t>(                    \
                _info, output, inputs, stream);                                     \
        case INFINI_DTYPE_F32:                                                      \
            return _device_info->template calculate<Op, float>(                     \
                _info, output, inputs, stream);                                     \
        default:                                                                    \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                                  \
        }                                                                           \
    }

/**
 * @brief Macro to generate unary operator implementation.
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
#define ELEMENTWISE_CPU_IMPL_UNARY(OP)                                              \
                                                                                    \
    Descriptor::~Descriptor() = default;                                            \
                                                                                    \
    infiniStatus_t Descriptor::create(                                              \
        infiniopHandle_t handle_,                                                   \
        Descriptor **desc_ptr,                                                      \
        infiniopTensorDescriptor_t out_desc,                                        \
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) {                   \
        auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);             \
        auto dtype = out_desc->dtype();                                             \
        const auto &x_desc = input_desc_vec.at(0);                                  \
        const auto &y_shape = out_desc->shape();                                    \
        const auto &x_shape = x_desc->shape();                                      \
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);                     \
        CHECK_SAME_SHAPE(y_shape, x_shape);                                         \
        CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec); \
        return INFINI_STATUS_SUCCESS;                                               \
    }                                                                               \
                                                                                    \
    infiniStatus_t Descriptor::calculate(                                           \
        void *workspace,                                                            \
        size_t workspace_size,                                                      \
        void *output,                                                               \
        std::vector<const void *> inputs,                                           \
        void *stream) const {                                                       \
        switch (_dtype) {                                                           \
        case INFINI_DTYPE_F16:                                                      \
            return _device_info->template calculate<Op, fp16_t>(                    \
                _info, output, inputs, stream);                                     \
        case INFINI_DTYPE_F32:                                                      \
            return _device_info->template calculate<Op, float>(                     \
                _info, output, inputs, stream);                                     \
        default:                                                                    \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                                  \
        }                                                                           \
    }

#endif // __INFINIOP_ELEMENTWISE_CPU_IMPL_H__
