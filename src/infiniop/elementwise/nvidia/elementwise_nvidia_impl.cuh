#ifndef __INFINIOP_ELEMENTWISE_NVIDIA_IMPL_CUH__
#define __INFINIOP_ELEMENTWISE_NVIDIA_IMPL_CUH__

#include "../../../utils/check.h"
#include "../../../utils/result.hpp"
#include "../../devices/nvidia/nvidia_common.cuh"
#include "elementwise_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

/**
 * @brief Generic implementation for elementwise NVIDIA/CUDA operators.
 *
 * This file provides a generic implementation template that can be used
 * by all binary and unary operators to reduce code duplication.
 *
 * Usage:
 *   #include "elementwise_nvidia_impl.cuh"
 *   namespace op::pow::nvidia {
 *       ELEMENTWISE_NVIDIA_IMPL_BINARY(pow)
 *   }
 *
 *   namespace op::sqrt::nvidia {
 *       ELEMENTWISE_NVIDIA_IMPL_UNARY(sqrt)
 *   }
 */

// =========================================================================
//  Internal Helpers (Private Macros to reduce duplication)
// =========================================================================

/**
 * @brief Common Calculate Switch Cases (F16 & F32)
 */
#define _IMPL_CALC_CASES_COMMON \
    case INFINI_DTYPE_F16: \
        return _device_info->calculate<256, cuda::Op, half>(_info, workspace, output, inputs, stream); \
    case INFINI_DTYPE_F32: \
        return _device_info->calculate<256, cuda::Op, float>(_info, workspace, output, inputs, stream);

/**
 * @brief Extended Calculate Switch Cases (Adds F64 & BF16)
 * Note: Order is F16, BF16, F32, F64 to match original implementation
 */
#define _IMPL_CALC_CASES_EXTENDED \
    case INFINI_DTYPE_F16: \
        return _device_info->calculate<256, cuda::Op, half>(_info, workspace, output, inputs, stream); \
    case INFINI_DTYPE_BF16: \
        return _device_info->calculate<256, cuda::Op, cuda_bfloat16>(_info, workspace, output, inputs, stream); \
    case INFINI_DTYPE_F32: \
        return _device_info->calculate<256, cuda::Op, float>(_info, workspace, output, inputs, stream); \
    case INFINI_DTYPE_F64: \
        return _device_info->calculate<256, cuda::Op, double>(_info, workspace, output, inputs, stream);

/**
 * @brief Generic Template for the Calculate method
 * @param CASES_MACRO The macro containing the switch cases to use
 */
#define _IMPL_CALCULATE_METHOD(CASES_MACRO) \
    infiniStatus_t Descriptor::calculate( \
        void *workspace, \
        size_t workspace_size, \
        void *output, \
        std::vector<const void *> inputs, \
        void *stream) const { \
        if (workspace_size < _workspace_size) { \
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE; \
        } \
        switch (_dtype) { \
            CASES_MACRO \
            default: \
                return INFINI_STATUS_BAD_TENSOR_DTYPE; \
        } \
    }

/**
 * @brief Generic Template for the Create method
 * @param SHAPE_CHECK_BLOCK Code block to execute for shape checking
 * @param ... Variadic arguments for allowed data types in CHECK_DTYPE
 */
#define _IMPL_CREATE_METHOD(SHAPE_CHECK_BLOCK, ...) \
    Descriptor::~Descriptor() = default; \
    infiniStatus_t Descriptor::create( \
        infiniopHandle_t handle_, \
        Descriptor **desc_ptr, \
        infiniopTensorDescriptor_t out_desc, \
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) { \
        auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_); \
        auto dtype = out_desc->dtype(); \
        const auto &out_shape = out_desc->shape(); \
        SHAPE_CHECK_BLOCK \
        CHECK_DTYPE(dtype, __VA_ARGS__); \
        CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec); \
        return INFINI_STATUS_SUCCESS; \
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
 *   namespace op::pow::nvidia {
 *       ELEMENTWISE_NVIDIA_IMPL_BINARY(pow)
 *   }
 */
#define ELEMENTWISE_NVIDIA_IMPL_BINARY(OP) \
    _IMPL_CREATE_METHOD( \
        const auto &a_desc = input_desc_vec.at(0); \
        const auto &b_desc = input_desc_vec.at(1); \
        const auto &a_shape = a_desc->shape(); \
        const auto &b_shape = b_desc->shape(); \
        CHECK_SAME_SHAPE(out_shape, a_shape, b_shape);, \
        INFINI_DTYPE_F16, INFINI_DTYPE_F32 \
    ) \
    _IMPL_CALCULATE_METHOD(_IMPL_CALC_CASES_COMMON)

/**
 * @brief Implementation for Unary Operators (F16, F32)
 *
 * This macro generates the Descriptor destructor, create, and calculate methods
 * for unary operators, using the generic implementation.
 *
 * Usage:
 *   namespace op::sqrt::nvidia {
 *       ELEMENTWISE_NVIDIA_IMPL_UNARY(sqrt)
 *   }
 */
#define ELEMENTWISE_NVIDIA_IMPL_UNARY(OP) \
    _IMPL_CREATE_METHOD( \
        const auto &x_desc = input_desc_vec.at(0); \
        const auto &x_shape = x_desc->shape(); \
        CHECK_SAME_SHAPE(out_shape, x_shape);, \
        INFINI_DTYPE_F16, INFINI_DTYPE_F32 \
    ) \
    _IMPL_CALCULATE_METHOD(_IMPL_CALC_CASES_COMMON)

/**
 * @brief Implementation for Unary Operators Extended (F16, F32, F64, BF16)
 *
 * This macro generates the Descriptor destructor, create, and calculate methods
 * for unary operators supporting F16, F32, F64, and BF16 data types.
 *
 * Usage:
 *   namespace op::exp::nvidia {
 *       ELEMENTWISE_NVIDIA_IMPL_UNARY_EXTENDED(exp)
 *   }
 */
#define ELEMENTWISE_NVIDIA_IMPL_UNARY_EXTENDED(OP) \
    _IMPL_CREATE_METHOD( \
        const auto &x_desc = input_desc_vec.at(0); \
        const auto &x_shape = x_desc->shape(); \
        CHECK_SAME_SHAPE(out_shape, x_shape);, \
        INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16 \
    ) \
    _IMPL_CALCULATE_METHOD(_IMPL_CALC_CASES_EXTENDED)

#endif // __INFINIOP_ELEMENTWISE_NVIDIA_IMPL_CUH__
