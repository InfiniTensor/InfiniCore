#include "tril_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../tensor.h"
#include "../../../../utils/custom_types.h"
#include <cstring>
#include <omp.h>

namespace op::tril::cpu {

// Simple tril kernel for 2D contiguous tensors
template<typename T>
static infiniStatus_t tril_kernel(
    T *output,
    const T *input,
    size_t rows,
    size_t cols,
    int diagonal) {
    
    #pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;
            // Keep lower triangular part (including diagonal offset)
            if (static_cast<int>(j) <= static_cast<int>(i) + diagonal) {
                output[idx] = input[idx];
            } else {
                output[idx] = T{}; // Zero out upper triangular part
            }
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}

// Inplace tril kernel for 2D contiguous tensors
template<typename T>
static infiniStatus_t tril_kernel_inplace(
    T *input_output,
    size_t rows,
    size_t cols,
    int diagonal) {
    
    #pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;
            // Zero out upper triangular part
            if (static_cast<int>(j) > static_cast<int>(i) + diagonal) {
                input_output[idx] = T{};
            }
            // Lower triangular part (including diagonal offset) remains unchanged
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int diagonal) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    // Check data types - 支持所有合法类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                INFINI_DTYPE_BOOL);

    // Check that input and output have same dtype
    if (input_desc->dtype() != output_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check dimensions - only support 2D tensors
    auto input_shape = input_desc->shape();
    auto output_shape = output_desc->shape();
    
    if (input_shape.size() != 2 || output_shape.size() != 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    if (input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    // Check that tensors are contiguous
    if (!input_desc->isContiguous() || !output_desc->isContiguous()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    auto desc = new Descriptor();
    desc->_input_desc = input_desc;
    desc->_output_desc = output_desc;
    desc->_diagonal = diagonal;
    desc->_handle = handle;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *input,
    void *stream) const {

    auto input_dtype = _input_desc->dtype();
    auto input_shape = _input_desc->shape();
    
    size_t rows = input_shape[0];
    size_t cols = input_shape[1];

    // Call kernel based on data type
    switch (input_dtype) {
    case INFINI_DTYPE_I8:
        return tril_kernel<int8_t>(
            static_cast<int8_t *>(output),
            static_cast<const int8_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_I16:
        return tril_kernel<int16_t>(
            static_cast<int16_t *>(output),
            static_cast<const int16_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_I32:
        return tril_kernel<int32_t>(
            static_cast<int32_t *>(output),
            static_cast<const int32_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_I64:
        return tril_kernel<int64_t>(
            static_cast<int64_t *>(output),
            static_cast<const int64_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_U8:
        return tril_kernel<uint8_t>(
            static_cast<uint8_t *>(output),
            static_cast<const uint8_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_U16:
        return tril_kernel<uint16_t>(
            static_cast<uint16_t *>(output),
            static_cast<const uint16_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_U32:
        return tril_kernel<uint32_t>(
            static_cast<uint32_t *>(output),
            static_cast<const uint32_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_U64:
        return tril_kernel<uint64_t>(
            static_cast<uint64_t *>(output),
            static_cast<const uint64_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_F16:
        return tril_kernel<fp16_t>(
            static_cast<fp16_t *>(output),
            static_cast<const fp16_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_F32:
        return tril_kernel<float>(
            static_cast<float *>(output),
            static_cast<const float *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_F64:
        return tril_kernel<double>(
            static_cast<double *>(output),
            static_cast<const double *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_BF16:
        return tril_kernel<bf16_t>(
            static_cast<bf16_t *>(output),
            static_cast<const bf16_t *>(input),
            rows, cols, _diagonal);
    case INFINI_DTYPE_BOOL:
        return tril_kernel<bool>(
            static_cast<bool *>(output),
            static_cast<const bool *>(input),
            rows, cols, _diagonal);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculateInplace(
    void *workspace,
    size_t workspace_size,
    void *input_output,
    void *stream) const {

    auto input_dtype = _input_desc->dtype();
    auto input_shape = _input_desc->shape();
    
    size_t rows = input_shape[0];
    size_t cols = input_shape[1];

    // Call inplace kernel based on data type
    switch (input_dtype) {
    case INFINI_DTYPE_I8:
        return tril_kernel_inplace<int8_t>(
            static_cast<int8_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_I16:
        return tril_kernel_inplace<int16_t>(
            static_cast<int16_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_I32:
        return tril_kernel_inplace<int32_t>(
            static_cast<int32_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_I64:
        return tril_kernel_inplace<int64_t>(
            static_cast<int64_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_U8:
        return tril_kernel_inplace<uint8_t>(
            static_cast<uint8_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_U16:
        return tril_kernel_inplace<uint16_t>(
            static_cast<uint16_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_U32:
        return tril_kernel_inplace<uint32_t>(
            static_cast<uint32_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_U64:
        return tril_kernel_inplace<uint64_t>(
            static_cast<uint64_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_F16:
        return tril_kernel_inplace<fp16_t>(
            static_cast<fp16_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_F32:
        return tril_kernel_inplace<float>(
            static_cast<float *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_F64:
        return tril_kernel_inplace<double>(
            static_cast<double *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_BF16:
        return tril_kernel_inplace<bf16_t>(
            static_cast<bf16_t *>(input_output),
            rows, cols, _diagonal);
    case INFINI_DTYPE_BOOL:
        return tril_kernel_inplace<bool>(
            static_cast<bool *>(input_output),
            rows, cols, _diagonal);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::tril::cpu