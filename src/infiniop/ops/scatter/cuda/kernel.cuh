#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace op::scatter::cuda {

// CUDA kernel for scatter operation
template<typename T, typename IndexT>
__global__ void scatter_kernel(
    T *output,
    const T *src,
    const IndexT *indices,
    size_t index_size,
    size_t index_size_unused,
    size_t output_size,
    size_t dim,
    const size_t *src_strides,
    const size_t *output_strides,
    const size_t *index_strides,
    const size_t *src_shape,
    const size_t *output_shape,
    size_t ndim) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < index_size && ndim <= 8) { // Add safety check for dimensions
        // idx is the linear index into the index tensor
        // Get target index from indices array
        IndexT target_index = indices[idx];
        
        // Calculate multi-dimensional coordinates from linear index
        size_t temp_idx = idx;
        size_t coords[8]; // Assuming max 8 dimensions
        
        // Convert linear index to multi-dimensional coordinates
        // This assumes index tensor has same shape as src tensor
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d] = temp_idx % src_shape[d];
            temp_idx /= src_shape[d];
        }
        
        // Calculate source offset
        size_t src_offset = 0;
        for (size_t d = 0; d < ndim; d++) {
            src_offset += coords[d] * src_strides[d];
        }
        
        // Boundary check for target_index
        if (target_index >= 0 && target_index < static_cast<IndexT>(output_shape[dim])) {
            // Calculate output coordinates (same as src coordinates except for dim)
            size_t output_coords[8];
            for (size_t d = 0; d < ndim; d++) {
                output_coords[d] = coords[d];
            }
            output_coords[dim] = static_cast<size_t>(target_index);
            
            // Calculate output position
            size_t output_offset = 0;
            for (size_t d = 0; d < ndim; d++) {
                output_offset += output_coords[d] * output_strides[d];
            }
            
            // Copy data
            output[output_offset] = src[src_offset];
        }
    }
}

// Launch scatter kernel
template<typename T, typename IndexT>
cudaError_t launch_scatter_kernel(
    T *output,
    const T *src,
    const IndexT *indices,
    size_t src_size,
    size_t index_size,
    size_t output_size,
    size_t dim,
    const size_t *src_strides,
    const size_t *output_strides,
    const size_t *index_strides,
    const size_t *src_shape,
    const size_t *output_shape,
    size_t ndim,
    cudaStream_t stream = 0) {
    
    const int block_size = 256;
    const int grid_size = (index_size + block_size - 1) / block_size;
    
    scatter_kernel<T, IndexT><<<grid_size, block_size, 0, stream>>>(
        output, src, indices, index_size, index_size, output_size, dim,
        src_strides, output_strides, index_strides,
        src_shape, output_shape, ndim
    );
    
    return cudaGetLastError();
}

} // namespace op::scatter::cuda