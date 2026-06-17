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

/**
 * @brief Macro to generate binary operator implementation for NVIDIA/CUDA.
 *
 * This macro generates the Descriptor destructor, create, and calculate methods
 * for binary operators, using the generic implementation.
 *
 * Usage:
 *   namespace op::pow::nvidia {
 *       ELEMENTWISE_NVIDIA_IMPL_BINARY(pow)
 *   }
 */
#define ELEMENTWISE_NVIDIA_IMPL_BINARY(OP)                                           \
                                                                                     \
    Descriptor::~Descriptor() = default;                                             \
                                                                                     \
    infiniStatus_t Descriptor::create(                                               \
        infiniopHandle_t handle_,                                                    \
        Descriptor **desc_ptr,                                                       \
        infiniopTensorDescriptor_t out_desc,                                         \
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) {                    \
        auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);           \
        auto dtype = out_desc->dtype();                                              \
        const auto &a_desc = input_desc_vec.at(0);                                   \
        const auto &b_desc = input_desc_vec.at(1);                                   \
        const auto &c_shape = out_desc->shape();                                     \
        const auto &a_shape = a_desc->shape();                                       \
        const auto &b_shape = b_desc->shape();                                       \
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);                      \
        CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);                                 \
        CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec); \
        return INFINI_STATUS_SUCCESS;                                                \
    }                                                                                \
                                                                                     \
    infiniStatus_t Descriptor::calculate(                                            \
        void *workspace,                                                             \
        size_t workspace_size,                                                       \
        void *output,                                                                \
        std::vector<const void *> inputs,                                            \
        void *stream) const {                                                        \
        if (workspace_size < _workspace_size) {                                      \
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;                             \
        }                                                                            \
        switch (_dtype) {                                                            \
        case INFINI_DTYPE_F16:                                                       \
            return _device_info->calculate<256, cuda::Op, half>(                     \
                _info, workspace, output, inputs, stream);                           \
        case INFINI_DTYPE_F32:                                                       \
            return _device_info->calculate<256, cuda::Op, float>(                    \
                _info, workspace, output, inputs, stream);                           \
        default:                                                                     \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                                   \
        }                                                                            \
    }

/**
 * @brief Macro to generate unary operator implementation for NVIDIA/CUDA.
 *
 * This macro generates the Descriptor destructor, create, and calculate methods
 * for unary operators, using the generic implementation.
 *
 * Usage:
 *   namespace op::sqrt::nvidia {
 *       ELEMENTWISE_NVIDIA_IMPL_UNARY(sqrt)
 *   }
 */
#define ELEMENTWISE_NVIDIA_IMPL_UNARY(OP)                                            \
                                                                                     \
    Descriptor::~Descriptor() = default;                                             \
                                                                                     \
    infiniStatus_t Descriptor::create(                                               \
        infiniopHandle_t handle_,                                                    \
        Descriptor **desc_ptr,                                                       \
        infiniopTensorDescriptor_t out_desc,                                         \
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) {                    \
        auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);           \
        auto dtype = out_desc->dtype();                                              \
        const auto &x_desc = input_desc_vec.at(0);                                   \
        const auto &y_shape = out_desc->shape();                                     \
        const auto &x_shape = x_desc->shape();                                       \
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);                      \
        CHECK_SAME_SHAPE(y_shape, x_shape);                                          \
        CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec); \
        return INFINI_STATUS_SUCCESS;                                                \
    }                                                                                \
                                                                                     \
    infiniStatus_t Descriptor::calculate(                                            \
        void *workspace,                                                             \
        size_t workspace_size,                                                       \
        void *output,                                                                \
        std::vector<const void *> inputs,                                            \
        void *stream) const {                                                        \
        if (workspace_size < _workspace_size) {                                      \
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;                             \
        }                                                                            \
        switch (_dtype) {                                                            \
        case INFINI_DTYPE_F16:                                                       \
            return _device_info->calculate<256, cuda::Op, half>(                     \
                _info, workspace, output, inputs, stream);                           \
        case INFINI_DTYPE_F32:                                                       \
            return _device_info->calculate<256, cuda::Op, float>(                    \
                _info, workspace, output, inputs, stream);                           \
        default:                                                                     \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                                   \
        }                                                                            \
    }

#endif // __INFINIOP_ELEMENTWISE_NVIDIA_IMPL_CUH__
