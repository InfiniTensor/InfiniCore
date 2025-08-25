#include "gather_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../tensor.h"
#include "../../../../utils/custom_types.h"
#include "../cuda/kernel.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace op::gather::nvidia {

// Use kernel from cuda folder
using op::gather::cuda::gather_kernel;

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int dim,
    infiniopTensorDescriptor_t index_desc) {

    if (!handle_ || !desc_ptr || !input_desc || !output_desc || !index_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto handle = static_cast<device::nvidia::Handle *>(handle_);
    
    // Get tensor shapes and strides
    auto input_shape = input_desc->shape();
    auto output_shape = output_desc->shape();
    auto index_shape = index_desc->shape();
    
    // Validate dimensions
    if (dim < 0 || dim >= static_cast<int>(input_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Check data types
    auto input_dtype = input_desc->dtype();
    auto output_dtype = output_desc->dtype();
    auto index_dtype = index_desc->dtype();
    
    if (input_dtype != output_dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    if (index_dtype != INFINI_DTYPE_I32 && index_dtype != INFINI_DTYPE_I64) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check that input data type is supported
    if (input_dtype != INFINI_DTYPE_F16 && input_dtype != INFINI_DTYPE_F32 && 
        input_dtype != INFINI_DTYPE_F64 && input_dtype != INFINI_DTYPE_BF16 &&
        input_dtype != INFINI_DTYPE_I8 && input_dtype != INFINI_DTYPE_I16 && 
        input_dtype != INFINI_DTYPE_I32 && input_dtype != INFINI_DTYPE_I64 &&
        input_dtype != INFINI_DTYPE_U8 && input_dtype != INFINI_DTYPE_U16 && 
        input_dtype != INFINI_DTYPE_U32 && input_dtype != INFINI_DTYPE_U64 &&
        input_dtype != INFINI_DTYPE_BOOL) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto desc = new Descriptor();
    desc->_input_desc = input_desc;
    desc->_output_desc = output_desc;
    desc->_index_desc = index_desc;
    desc->_dim = dim;
    desc->_input_shape = input_shape;
    desc->_output_shape = output_shape;
    desc->_input_strides = input_desc->getByteStrides();
    desc->_output_strides = output_desc->getByteStrides();
    desc->_index_strides = index_desc->getByteStrides();
    desc->_dtype = input_dtype;
    desc->_index_dtype = index_dtype;
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
    const void *input,
    const void *index,
    void *stream) const {

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    // Calculate total output elements
    size_t total_elements = 1;
    for (size_t d = 0; d < _output_shape.size(); d++) {
        total_elements *= _output_shape[d];
    }
    
    if (total_elements == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    
    // Allocate device memory for shape and stride arrays
    size_t *d_output_shape;
    ptrdiff_t *d_input_strides, *d_output_strides, *d_index_strides;
    
    cudaMalloc(&d_output_shape, _output_shape.size() * sizeof(size_t));
    cudaMalloc(&d_input_strides, _input_strides.size() * sizeof(ptrdiff_t));
    cudaMalloc(&d_output_strides, _output_strides.size() * sizeof(ptrdiff_t));
    cudaMalloc(&d_index_strides, _index_strides.size() * sizeof(ptrdiff_t));
    
    cudaMemcpy(d_output_shape, _output_shape.data(), _output_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_strides, _input_strides.data(), _input_strides.size() * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_strides, _output_strides.data(), _output_strides.size() * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_strides, _index_strides.data(), _index_strides.size() * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    size_t input_dim_size = _input_shape[_dim];
    size_t ndim = _output_shape.size();

    // Call kernel based on data type and index type
    if (_index_dtype == INFINI_DTYPE_I32) {
        switch (_dtype) {
        case INFINI_DTYPE_F16:
            gather_kernel<fp16_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<fp16_t *>(output),
                static_cast<const fp16_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_F32:
            gather_kernel<float, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<float *>(output),
                static_cast<const float *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_F64:
            gather_kernel<double, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<double *>(output),
                static_cast<const double *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_I32:
            gather_kernel<int32_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<int32_t *>(output),
                static_cast<const int32_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_I64:
            gather_kernel<int64_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<int64_t *>(output),
                static_cast<const int64_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_BF16:
            gather_kernel<bf16_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<bf16_t *>(output),
                static_cast<const bf16_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_I8:
            gather_kernel<int8_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<int8_t *>(output),
                static_cast<const int8_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_I16:
            gather_kernel<int16_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<int16_t *>(output),
                static_cast<const int16_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_U8:
            gather_kernel<uint8_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<uint8_t *>(output),
                static_cast<const uint8_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_U16:
            gather_kernel<uint16_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<uint16_t *>(output),
                static_cast<const uint16_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_U32:
            gather_kernel<uint32_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<uint32_t *>(output),
                static_cast<const uint32_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_U64:
            gather_kernel<uint64_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<uint64_t *>(output),
                static_cast<const uint64_t *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_BOOL:
            gather_kernel<bool, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<bool *>(output),
                static_cast<const bool *>(input),
                static_cast<const int32_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        default:
            cudaFree(d_output_shape);
            cudaFree(d_input_strides);
            cudaFree(d_output_strides);
            cudaFree(d_index_strides);
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_index_dtype == INFINI_DTYPE_I64) {
        switch (_dtype) {
        case INFINI_DTYPE_F16:
            gather_kernel<fp16_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<fp16_t *>(output),
                static_cast<const fp16_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_F32:
            gather_kernel<float, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<float *>(output),
                static_cast<const float *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_F64:
            gather_kernel<double, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<double *>(output),
                static_cast<const double *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_I32:
            gather_kernel<int32_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<int32_t *>(output),
                static_cast<const int32_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_I64:
            gather_kernel<int64_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<int64_t *>(output),
                static_cast<const int64_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_BF16:
            gather_kernel<bf16_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<bf16_t *>(output),
                static_cast<const bf16_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_I8:
            gather_kernel<int8_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<int8_t *>(output),
                static_cast<const int8_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_I16:
            gather_kernel<int16_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<int16_t *>(output),
                static_cast<const int16_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_U8:
            gather_kernel<uint8_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<uint8_t *>(output),
                static_cast<const uint8_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_U16:
            gather_kernel<uint16_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<uint16_t *>(output),
                static_cast<const uint16_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_U32:
            gather_kernel<uint32_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<uint32_t *>(output),
                static_cast<const uint32_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_U64:
            gather_kernel<uint64_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<uint64_t *>(output),
                static_cast<const uint64_t *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        case INFINI_DTYPE_BOOL:
            gather_kernel<bool, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<bool *>(output),
                static_cast<const bool *>(input),
                static_cast<const int64_t *>(index),
                total_elements, _dim, input_dim_size,
                d_output_shape, d_input_strides, d_output_strides, d_index_strides, ndim);
            break;
        default:
            cudaFree(d_output_shape);
            cudaFree(d_input_strides);
            cudaFree(d_output_strides);
            cudaFree(d_index_strides);
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        cudaFree(d_output_shape);
        cudaFree(d_input_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_output_shape);
        cudaFree(d_input_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    // Free device memory
    cudaFree(d_output_shape);
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    cudaFree(d_index_strides);
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gather::nvidia