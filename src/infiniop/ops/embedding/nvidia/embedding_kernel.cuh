#ifndef __EMBEDDING_CUDA_KERNEL_CUH__
#define __EMBEDDING_CUDA_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cuda_runtime.h>

namespace op::embedding::nvidia {

template <typename T, typename IndexType>
INFINIOP_CUDA_KERNEL embeddingKernel(
    T *output,
    const IndexType *indices,
    const T *weight,
    size_t num_indices,
    size_t embedding_dim,
    size_t vocab_size) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        // Get the index value
        IndexType index_val = indices[idx];
        
        // Bounds check - handle negative indices gracefully
        if (index_val >= 0 && static_cast<size_t>(index_val) < vocab_size) {
            // Copy embedding vector from weight to output
            const T *src = weight + static_cast<size_t>(index_val) * embedding_dim;
            T *dst = output + idx * embedding_dim;
            
            // Copy embedding_dim elements
            // Use vectorized copy for better performance when possible
            size_t i = 0;
            // Copy in chunks of 4 for better memory bandwidth utilization
            for (; i + 4 <= embedding_dim; i += 4) {
                dst[i] = src[i];
                dst[i + 1] = src[i + 1];
                dst[i + 2] = src[i + 2];
                dst[i + 3] = src[i + 3];
            }
            // Copy remaining elements
            for (; i < embedding_dim; ++i) {
                dst[i] = src[i];
            }
        }
    }
}

} // namespace op::embedding::nvidia

#endif // __EMBEDDING_CUDA_KERNEL_CUH__
