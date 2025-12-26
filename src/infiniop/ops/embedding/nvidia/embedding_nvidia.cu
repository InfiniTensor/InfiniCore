#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "../../../../utils.h"
#include "embedding_kernel.cuh"
#include "embedding_nvidia.cuh"
#include <cuda_runtime.h>

namespace op::embedding::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc) {

    auto handle_nvidia = reinterpret_cast<device::nvidia::Handle *>(handle);
    auto input_shape = input_desc->shape();
    auto weight_shape = weight_desc->shape();
    
    // Validate shapes
    CHECK_OR_RETURN(weight_shape.size() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->shape().size() == input_shape.size() + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    
    // Check output shape matches input shape + embedding_dim
    auto output_shape = output_desc->shape();
    size_t embedding_dim = weight_shape[1];
    CHECK_OR_RETURN(output_shape.back() == embedding_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);
    
    for (size_t i = 0; i < input_shape.size(); ++i) {
        CHECK_OR_RETURN(output_shape[i] == input_shape[i], INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    
    // Validate dtypes
    auto input_dtype = input_desc->dtype();
    auto weight_dtype = weight_desc->dtype();
    CHECK_OR_RETURN(input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(weight_dtype == INFINI_DTYPE_F32 || weight_dtype == INFINI_DTYPE_F16 ||
                    weight_dtype == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->dtype() == weight_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    
    // Calculate number of indices (supporting batch dimension)
    size_t num_indices = 1;
    for (auto dim : input_shape) {
        num_indices *= dim;
    }
    
    size_t vocab_size = weight_shape[0];
    
    *desc_ptr = new Descriptor(
        num_indices,
        embedding_dim,
        vocab_size,
        input_dtype,
        weight_dtype,
        new Opaque{handle_nvidia->internal()},
        handle->device,
        handle->device_id);
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    const void *input,
    const void *weight,
    void *stream) const {
    
    if (_num_indices == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr size_t BLOCK_SIZE = 256;
    size_t grid_size = (_num_indices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel based on dtypes
    if (_input_dtype == INFINI_DTYPE_I32) {
        const int32_t *indices_ptr = reinterpret_cast<const int32_t *>(input);
        
        if (_weight_dtype == INFINI_DTYPE_F32) {
            embeddingKernel<float, int32_t><<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(output),
                indices_ptr,
                reinterpret_cast<const float *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_F16) {
            embeddingKernel<half, int32_t><<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(output),
                indices_ptr,
                reinterpret_cast<const half *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_BF16) {
            embeddingKernel<cuda_bfloat16, int32_t><<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(output),
                indices_ptr,
                reinterpret_cast<const cuda_bfloat16 *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_input_dtype == INFINI_DTYPE_I64) {
        const int64_t *indices_ptr = reinterpret_cast<const int64_t *>(input);
        
        if (_weight_dtype == INFINI_DTYPE_F32) {
            embeddingKernel<float, int64_t><<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(output),
                indices_ptr,
                reinterpret_cast<const float *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_F16) {
            embeddingKernel<half, int64_t><<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(output),
                indices_ptr,
                reinterpret_cast<const half *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_BF16) {
            embeddingKernel<cuda_bfloat16, int64_t><<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(output),
                indices_ptr,
                reinterpret_cast<const cuda_bfloat16 *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::embedding::nvidia
