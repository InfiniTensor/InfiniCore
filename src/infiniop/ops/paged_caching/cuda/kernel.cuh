#ifndef __PAGED_CACHING_KERNEL_CUH__
#define __PAGED_CACHING_KERNEL_CUH__

#include <cuda_fp16.h>

//================================================================================
// Paged Caching Operator CUDA Kernel
//
// This kernel implements the "paged_caching" operation, which copies Key and Value
// vectors from a contiguous source tensor into a paged, non-contiguous KV Cache.
//
// Design Principles:
// 1. Token-Centric Parallelism: A 1D grid of `num_tokens` is launched. Each CUDA
//    block is responsible for caching one full token (all its heads).
// 2. Coalesced Memory Access: This grid strategy ensures that threads within a
//    block read a large, contiguous chunk of memory from the source tensors,
//    maximizing memory bandwidth utilization.
// 3. Vectorization: The copy operation is vectorized to further enhance memory
//    throughput, processing multiple data elements in a single instruction.
//================================================================================

namespace op::paged_caching::cuda {

template <
    typename Tdata,   // Data type of the tensors (e.g., half, __nv_bfloat16)
    int NUM_THREADS   // Number of threads per block, configured at launch time
>
__device__ void pagedCachingKernel(
    // ----- Output Tensors -----
    Tdata*  k_cache_ptr,         // Pointer to the destination K cache pool
    Tdata*  v_cache_ptr,         // Pointer to the destination V cache pool
    // ----- Input Tensors -----
    const Tdata*  k_ptr,         // Pointer to the source Keys, shape [ntok, nkvh, dh]
    const Tdata*  v_ptr,         // Pointer to the source Values, shape [ntok, nkvh, dh]
    const int*  slot_mapping_ptr, // Pointer to the slot mapping, shape [ntok]
    // ----- Metadata -----
    const int num_heads,                      // Number of key/value heads (nkvh)
    const int head_size,                      // Dimension of each head (dh)
    const int block_size,                     // Number of tokens per block in the KV cache
    // ----- Stride Information -----
    const ptrdiff_t k_src_stride,             // Stride between tokens in the source K tensor
    const ptrdiff_t v_src_stride,             // Stride between tokens in the source V tensor
    const ptrdiff_t k_cache_block_stride,     // Stride between blocks in the K cache pool
    const ptrdiff_t v_cache_block_stride      // Stride between blocks in the V cache pool
) {
    //================================================================================
    // 1. Identify Work Unit & Calculate Addresses
    //================================================================================
    
    // Each block processes one token.
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;

    // Retrieve the destination slot for the current token.
    const int slot_idx = slot_mapping_ptr[token_idx];

    // Handle padding: if slot_idx is negative, this token is padding and should be ignored.
    if (slot_idx < 0) {
        return;
    }

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("[Block %d, Thread %d] Debug Start\n", blockIdx.x, threadIdx.x);
    //     printf("  - token_idx: %d\n", token_idx);
    //     printf("  - slot_idx from mapping: %d\n", slot_idx);
    //     printf("  - Metadata: num_heads=%d, head_size=%d, block_size=%d\n", num_heads, head_size, block_size);
    // }

    // Calculate the physical block index and the offset within that block.
    const int physical_block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    // Calculate base pointers for source and destination for this specific token.
    const Tdata* k_src_head_ptr = k_ptr + token_idx * k_src_stride + head_idx * head_size;
    const Tdata* v_src_head_ptr = v_ptr + token_idx * v_src_stride + head_idx * head_size;


    // Destination pointer calculation assumes a [num_blocks, block_size, num_heads, head_size] layout.
    // We point to the beginning of the memory region for this token's slot.
    const ptrdiff_t cache_head_stride = block_size * head_size;
    
    Tdata* k_cache_block_base_ptr = k_cache_ptr + physical_block_idx * k_cache_block_stride;
    Tdata* k_dst_head_ptr = k_cache_block_base_ptr + head_idx * cache_head_stride + block_offset * head_size;

    Tdata* v_cache_block_base_ptr = v_cache_ptr + physical_block_idx * v_cache_block_stride;
    Tdata* v_dst_head_ptr = v_cache_block_base_ptr + head_idx * cache_head_stride + block_offset * head_size;


    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("[Block %d, Thread %d] Address Calculation\n", blockIdx.x, threadIdx.x);
    //     printf("  - physical_block_idx: %d\n", physical_block_idx);
    //     printf("  - block_offset: %d\n", block_offset);
    //     printf("  - k_src_stride: %ld, v_src_stride: %ld\n", k_src_stride, v_src_stride);
    //     printf("  - k_cache_block_stride: %ld, v_cache_block_stride: %ld\n", k_cache_block_stride, v_cache_block_stride);
    //     printf("  - Calculated dst_token_offset_k: %d\n", dst_token_offset_k);
    //     printf("  - Source Ptr (k): %p\n", k_src_token_ptr);
    //     printf("  - Dest Ptr (k_cache): %p\n", k_cache_dst_ptr);
    // }


    // //================================================================================
    // // 2. Perform Vectorized Data Copy
    // //================================================================================

    // // Total number of elements to copy for one token (all heads).
    // const int total_elements_per_token = num_heads * head_size;

    // // Use vectorization to copy data more efficiently.
    // // For Tdata=half (2 bytes), float4 (16 bytes) can process 8 elements at once.
    // constexpr int VEC_SIZE = sizeof(float4) / sizeof(Tdata);

    // // Cast pointers to the vectorized type.
    // const float4* k_src_vec_ptr = reinterpret_cast<const float4*>(k_src_token_ptr);
    // const float4* v_src_vec_ptr = reinterpret_cast<const float4*>(v_src_token_ptr);
    // float4* k_cache_dst_vec_ptr = reinterpret_cast<float4*>(k_cache_dst_ptr);
    // float4* v_cache_dst_vec_ptr = reinterpret_cast<float4*>(v_cache_dst_ptr);

    // // if (blockIdx.x == 0 && threadIdx.x == 0) {
    // //     printf("[Block %d, Thread %d] Vectorized Copy Start\n", blockIdx.x, threadIdx.x);
    // //     printf("  - Total elements per token: %d\n", total_elements_per_token);
    // //     printf("  - Vector size: %d\n", VEC_SIZE);
    // //     printf("  - float4 size: %d\n", sizeof(float4));
    // //     printf("  - Tdata size: %d\n", sizeof(Tdata));
    // //     // printf("  - Vector size: %d\n", k_src_vec_ptr[i]);
    // // }

    // // Each thread copies one vector (VEC_SIZE elements) per iteration.
    // // The loop iterates over the vectorized chunks of data for the token.
    // for (int i = threadIdx.x; i < total_elements_per_token / VEC_SIZE; i += NUM_THREADS) {

    //     k_cache_dst_vec_ptr[i] = k_src_vec_ptr[i];
    //     v_cache_dst_vec_ptr[i] = v_src_vec_ptr[i];
    // }
    // }

    //================================================================================
    // 2. Perform Element-wise Data Copy (Safe, Non-Vectorized)
    //================================================================================
    for (int i = threadIdx.x; i < head_size; i += NUM_THREADS) {
        k_dst_head_ptr[i] = k_src_head_ptr[i];
        v_dst_head_ptr[i] = v_src_head_ptr[i];
    }
}

} // namespace op::paged_caching::cuda

#endif // __PAGED_CACHING_KERNEL_CUH__