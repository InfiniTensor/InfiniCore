#ifndef __INDEX_COPY_INPLACE_CUDA_H__
#define __INDEX_COPY_INPLACE_CUDA_H__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace op::index_copy_inplace::cuda {

template<typename T, typename IndexT>
__global__ void index_copy_inplace_kernel(
    const T *input_data,
    T *output_data,
    const IndexT *index_data,
    const int *input_shape,
    const int *output_shape,
    const int *index_shape,
    const int *input_strides,
    const int *output_strides,
    const int *index_strides,
    int dim,
    int ndim,
    size_t total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Calculate input coordinates from linear index
    int in_coords[8]; // Support up to 8D tensors
    int temp = idx;
    for (int d = ndim - 1; d >= 0; --d) {
        in_coords[d] = temp % input_shape[d];
        temp /= input_shape[d];
    }
    
    // Get the index value for the current position in the specified dimension
    int src_idx = in_coords[dim];
    if (src_idx >= index_shape[0]) {
        return; // Skip if source index is out of bounds
    }
    
    // Get the target index from the index tensor
    IndexT target_idx = index_data[src_idx * index_strides[0]];
    
    // Check bounds for target index
    if (target_idx < 0 || target_idx >= output_shape[dim]) {
        return; // Skip out of bounds target indices
    }
    
    // Calculate output coordinates (copy input coords and modify dim)
    int out_coords[8];
    for (int d = 0; d < ndim; ++d) {
        out_coords[d] = in_coords[d];
    }
    out_coords[dim] = static_cast<int>(target_idx);
    
    // Calculate input offset
    size_t in_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        in_offset += in_coords[d] * input_strides[d];
    }
    
    // Calculate output offset
    size_t out_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        out_offset += out_coords[d] * output_strides[d];
    }
    
    // Copy the value from input to output at the indexed position
    output_data[out_offset] = input_data[in_offset];
}

} // namespace op::index_copy_inplace::cuda

#endif // __INDEX_COPY_INPLACE_CUDA_H__