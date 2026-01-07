#ifndef __EMBEDDING_CUDA_KERNEL_CUH__
#define __EMBEDDING_CUDA_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace op::embedding::nvidia {

// Helper function to check memory alignment
__forceinline__ __device__ bool is_aligned(const void *ptr, size_t alignment) {
    // Use size_t for pointer arithmetic in device code (more compatible)
    return (reinterpret_cast<size_t>(ptr) % alignment == 0);
}

// Vectorized copy for float type using float4
template <typename IndexType>
__forceinline__ __device__ void copyVectorizedFloat4(
    float *__restrict__ dst,
    const float *__restrict__ src,
    size_t embedding_dim) {
    // Use float4 for vectorized access (16 bytes, 4 floats)
    const float4 *src_vec = reinterpret_cast<const float4 *>(src);
    float4 *dst_vec = reinterpret_cast<float4 *>(dst);
    size_t vec_count = embedding_dim / 4;

    // Vectorized copy using __ldg for read-only weight
    for (size_t i = 0; i < vec_count; ++i) {
        dst_vec[i] = __ldg(&src_vec[i]);
    }

    // Copy remaining elements
    size_t remaining = embedding_dim % 4;
    if (remaining > 0) {
        size_t offset = vec_count * 4;
        for (size_t i = 0; i < remaining; ++i) {
            dst[offset + i] = __ldg(&src[offset + i]);
        }
    }
}

// Vectorized copy for float type using float2 (fallback when not aligned to 16 bytes)
template <typename IndexType>
__forceinline__ __device__ void copyVectorizedFloat2(
    float *__restrict__ dst,
    const float *__restrict__ src,
    size_t embedding_dim) {
    // Use float2 for vectorized access (8 bytes, 2 floats)
    const float2 *src_vec = reinterpret_cast<const float2 *>(src);
    float2 *dst_vec = reinterpret_cast<float2 *>(dst);
    size_t vec_count = embedding_dim / 2;

    // Vectorized copy using __ldg for read-only weight
    for (size_t i = 0; i < vec_count; ++i) {
        dst_vec[i] = __ldg(&src_vec[i]);
    }

    // Copy remaining element if odd
    if (embedding_dim % 2 != 0) {
        dst[embedding_dim - 1] = __ldg(&src[embedding_dim - 1]);
    }
}

// Vectorized copy for half type using half2
template <typename IndexType>
__forceinline__ __device__ void copyVectorizedHalf2(
    half *__restrict__ dst,
    const half *__restrict__ src,
    size_t embedding_dim) {
    // Use half2 for vectorized access (4 bytes, 2 halfs)
    const half2 *src_vec = reinterpret_cast<const half2 *>(src);
    half2 *dst_vec = reinterpret_cast<half2 *>(dst);
    size_t vec_count = embedding_dim / 2;

    // Vectorized copy using __ldg for read-only weight
    for (size_t i = 0; i < vec_count; ++i) {
        dst_vec[i] = __ldg(&src_vec[i]);
    }

    // Copy remaining element if odd
    if (embedding_dim % 2 != 0) {
        dst[embedding_dim - 1] = __ldg(&src[embedding_dim - 1]);
    }
}

// Vectorized copy for bfloat16 type using bfloat162
template <typename IndexType>
__forceinline__ __device__ void copyVectorizedBFloat162(
    cuda_bfloat16 *__restrict__ dst,
    const cuda_bfloat16 *__restrict__ src,
    size_t embedding_dim) {
    // Use bfloat162 for vectorized access (4 bytes, 2 bfloat16s)
    const cuda_bfloat162 *src_vec = reinterpret_cast<const cuda_bfloat162 *>(src);
    cuda_bfloat162 *dst_vec = reinterpret_cast<cuda_bfloat162 *>(dst);
    size_t vec_count = embedding_dim / 2;

    // Vectorized copy using __ldg for read-only weight
    for (size_t i = 0; i < vec_count; ++i) {
        dst_vec[i] = __ldg(&src_vec[i]);
    }

    // Copy remaining element if odd
    if (embedding_dim % 2 != 0) {
        dst[embedding_dim - 1] = __ldg(&src[embedding_dim - 1]);
    }
}

// Scalar copy fallback with __ldg optimization
template <typename T, typename IndexType>
__forceinline__ __device__ void copyScalar(
    T *__restrict__ dst,
    const T *__restrict__ src,
    size_t embedding_dim) {
    // Scalar copy with __ldg for read-only weight
    for (size_t i = 0; i < embedding_dim; ++i) {
        dst[i] = __ldg(&src[i]);
    }
}

template <typename T, typename IndexType>
INFINIOP_CUDA_KERNEL embeddingKernel(
    T *__restrict__ output,
    const IndexType *__restrict__ indices,
    const T *__restrict__ weight,
    size_t num_indices,
    size_t embedding_dim,
    size_t vocab_size) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_indices) {
        // Get the index value
        IndexType index_val = __ldg(&indices[idx]);

        // Bounds check - handle negative indices gracefully
        if (index_val >= 0 && static_cast<size_t>(index_val) < vocab_size) {
            // Copy embedding vector from weight to output
            const T *src = weight + static_cast<size_t>(index_val) * embedding_dim;
            T *dst = output + idx * embedding_dim;

            // Choose optimal copy strategy based on type and alignment
            if constexpr (std::is_same_v<T, float>) {
                // Check alignment for float4 (16 bytes)
                bool aligned_16 = is_aligned(src, 16) && is_aligned(dst, 16);
                if (aligned_16 && embedding_dim >= 4 && embedding_dim % 4 == 0) {
                    copyVectorizedFloat4<IndexType>(dst, src, embedding_dim);
                } else if (embedding_dim >= 2 && embedding_dim % 2 == 0) {
                    // Try float2 if not aligned to 16 bytes
                    copyVectorizedFloat2<IndexType>(dst, src, embedding_dim);
                } else {
                    copyScalar<T, IndexType>(dst, src, embedding_dim);
                }
            } else if constexpr (std::is_same_v<T, half>) {
                // Use half2 for vectorized access
                if (embedding_dim >= 2 && embedding_dim % 2 == 0) {
                    copyVectorizedHalf2<IndexType>(dst, src, embedding_dim);
                } else {
                    copyScalar<T, IndexType>(dst, src, embedding_dim);
                }
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                // Use bfloat162 for vectorized access
                if (embedding_dim >= 2 && embedding_dim % 2 == 0) {
                    copyVectorizedBFloat162<IndexType>(dst, src, embedding_dim);
                } else {
                    copyScalar<T, IndexType>(dst, src, embedding_dim);
                }
            } else {
                // Fallback to scalar copy with __ldg
                copyScalar<T, IndexType>(dst, src, embedding_dim);
            }
        }
    }
}

} // namespace op::embedding::nvidia

#endif // __EMBEDDING_CUDA_KERNEL_CUH__
