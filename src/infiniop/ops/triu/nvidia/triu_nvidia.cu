#include "triu_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../tensor.h"
#include "../../../../utils/custom_types.h"
#include "../cuda/kernel.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace op::triu::nvidia {

// Use kernels from cuda folder
using op::triu::cuda::triu_kernel;
using op::triu::cuda::triu_kernel_inplace;

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int diagonal) {

    if (!handle_ || !desc_ptr || !input_desc || !output_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto handle = static_cast<device::nvidia::Handle *>(handle_);
    
    // Check tensor dimensions (should be 2D)
    auto input_shape = input_desc->shape();
    auto output_shape = output_desc->shape();
    
    if (input_shape.size() != 2 || output_shape.size() != 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    if (input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    // Check data types match
    if (input_desc->dtype() != output_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto desc = new Descriptor();
    desc->_input_desc = input_desc;
    desc->_output_desc = output_desc;
    desc->_diagonal = diagonal;
    desc->_handle = handle;
    desc->device_type = handle_->device;
    desc->device_id = handle_->device_id;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *input,
    void *stream) const {

    auto input_shape = _input_desc->shape();
    size_t rows = input_shape[0];
    size_t cols = input_shape[1];
    size_t total_elements = rows * cols;
    auto input_dtype = _input_desc->dtype();
    
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    // Calculate grid and block dimensions
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    // Call kernel based on data type
    switch (input_dtype) {
    case INFINI_DTYPE_I8:
        triu_kernel<int8_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int8_t *>(output),
            static_cast<const int8_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_I16:
        triu_kernel<int16_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int16_t *>(output),
            static_cast<const int16_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_I32:
        triu_kernel<int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int32_t *>(output),
            static_cast<const int32_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_I64:
        triu_kernel<int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int64_t *>(output),
            static_cast<const int64_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_U8:
        triu_kernel<uint8_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<uint8_t *>(output),
            static_cast<const uint8_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_U16:
        triu_kernel<uint16_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<uint16_t *>(output),
            static_cast<const uint16_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_U32:
        triu_kernel<uint32_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<uint32_t *>(output),
            static_cast<const uint32_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_U64:
        triu_kernel<uint64_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<uint64_t *>(output),
            static_cast<const uint64_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_F16:
        triu_kernel<fp16_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<fp16_t *>(output),
            static_cast<const fp16_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_F32:
        triu_kernel<float><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<float *>(output),
            static_cast<const float *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_F64:
        triu_kernel<double><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<double *>(output),
            static_cast<const double *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_BF16:
        triu_kernel<bf16_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<bf16_t *>(output),
            static_cast<const bf16_t *>(input),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_BOOL:
        triu_kernel<bool><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<bool *>(output),
            static_cast<const bool *>(input),
            rows, cols, _diagonal);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculateInplace(
    void *workspace,
    size_t workspace_size,
    void *input_output,
    void *stream) const {

    auto input_shape = _input_desc->shape();
    size_t rows = input_shape[0];
    size_t cols = input_shape[1];
    size_t total_elements = rows * cols;
    auto input_dtype = _input_desc->dtype();
    
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    // Calculate grid and block dimensions
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    // Call kernel based on data type
    switch (input_dtype) {
    case INFINI_DTYPE_I8:
        triu_kernel_inplace<int8_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int8_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_I16:
        triu_kernel_inplace<int16_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int16_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_I32:
        triu_kernel_inplace<int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int32_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_I64:
        triu_kernel_inplace<int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int64_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_U8:
        triu_kernel_inplace<uint8_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<uint8_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_U16:
        triu_kernel_inplace<uint16_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<uint16_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_U32:
        triu_kernel_inplace<uint32_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<uint32_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_U64:
        triu_kernel_inplace<uint64_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<uint64_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_F16:
        triu_kernel_inplace<fp16_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<fp16_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_F32:
        triu_kernel_inplace<float><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<float *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_F64:
        triu_kernel_inplace<double><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<double *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_BF16:
        triu_kernel_inplace<bf16_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<bf16_t *>(input_output),
            rows, cols, _diagonal);
        break;
    case INFINI_DTYPE_BOOL:
        triu_kernel_inplace<bool><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<bool *>(input_output),
            rows, cols, _diagonal);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::triu::nvidia