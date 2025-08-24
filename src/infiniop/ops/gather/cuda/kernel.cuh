#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>

namespace op::gather::cuda {

// CUDA kernel for gather operation
template<typename T, typename IndexT>
__global__ void gather_kernel(
    T *output,
    const T *input,
    const IndexT *index,
    size_t total_elements,
    size_t dim,
    size_t input_dim_size,
    const size_t *output_shape,
    const ptrdiff_t *input_strides,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *index_strides,
    size_t ndim) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Convert linear index to multi-dimensional coordinates
        size_t temp_idx = idx;
        size_t coords[8]; // Support up to 8 dimensions
        
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d] = temp_idx % output_shape[d];
            temp_idx /= output_shape[d];
        }
        
        // Calculate output offset
        size_t output_offset = 0;
        for (size_t d = 0; d < ndim; d++) {
            output_offset += coords[d] * output_strides[d];
        }
        
        // Calculate index offset
        size_t index_offset = 0;
        for (size_t d = 0; d < ndim; d++) {
            index_offset += coords[d] * index_strides[d];
        }
        
        // Get gather index and perform bounds check
        IndexT gather_index = *((const IndexT*)((const char*)index + index_offset));
        if (gather_index >= 0 && gather_index < static_cast<IndexT>(input_dim_size)) {
            // Calculate input offset
            size_t input_offset = 0;
            for (size_t d = 0; d < ndim; d++) {
                if (d == dim) {
                    input_offset += gather_index * input_strides[d];
                } else {
                    input_offset += coords[d] * input_strides[d];
                }
            }
            
            // Copy data using element-wise access
            size_t input_elem_idx = input_offset / sizeof(T);
            size_t output_elem_idx = output_offset / sizeof(T);
            output[output_elem_idx] = input[input_elem_idx];
        }
    }
}

} // namespace op::gather::cuda