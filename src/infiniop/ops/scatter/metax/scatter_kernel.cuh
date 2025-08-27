#ifndef __SCATTER_KERNEL_CUH__
#define __SCATTER_KERNEL_CUH__

template <typename T, typename IndexT>
__device__ void scatterKernel(
    T *output,
    const T *input,
    const IndexT *index,
    const T *src,
    const size_t *src_shape,
    const size_t *output_shape,
    const size_t *src_strides,
    const size_t *output_strides,
    const size_t *index_strides,
    size_t dim,
    size_t total_src_elements,
    size_t ndim) {
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_src_elements) {
        return;
    }
    
    // Convert linear index to multi-dimensional coordinates
    size_t temp_idx = tid;
    size_t coords[8]; // Assume max 8 dimensions
    
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_idx % src_shape[d];
        temp_idx /= src_shape[d];
    }
    
    // Calculate src offset
    size_t src_offset = 0;
    for (size_t d = 0; d < ndim; d++) {
        src_offset += coords[d] * src_strides[d];
    }
    
    // Calculate index offset
    size_t index_offset = 0;
    for (size_t d = 0; d < ndim; d++) {
        index_offset += coords[d] * index_strides[d];
    }
    
    // Get the index value
    IndexT idx_val = index[index_offset];
    
    // Check bounds
    if (idx_val < 0 || idx_val >= static_cast<IndexT>(output_shape[dim])) {
        return; // Skip out-of-bounds indices
    }
    
    // Calculate output coordinates (same as src coordinates except for dim)
    coords[dim] = static_cast<size_t>(idx_val);
    
    // Calculate output offset
    size_t output_offset = 0;
    for (size_t d = 0; d < ndim; d++) {
        output_offset += coords[d] * output_strides[d];
    }
    
    // Perform scatter operation
    output[output_offset] = src[src_offset];
}

#endif