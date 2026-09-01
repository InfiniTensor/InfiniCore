#ifndef __PAGED_CACHING_FP8_KERNEL_CUH__
#define __PAGED_CACHING_FP8_KERNEL_CUH__

//================================================================================
// Paged Caching FP8(E4M3) Quantizing Write Kernel
//
// Same grid/slot semantics as the plain copy kernel in kernel.cuh, but each
// block additionally performs dynamic per-token-per-head quantization:
//   amax  = max(|x[0:head_size]|) over the head_dim vector
//   scale = amax / 448           (scale = 1 when amax == 0)
//   q     = e4m3_encode(x / scale)
// The scale is written to k_scale/v_scale at [block, kv_head, block_offset]
// and the encoded byte to the F8 cache. amax==0 yields scale=1 and all-zero
// codes (encode(0)==0).
//================================================================================

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

namespace op::paged_caching::cuda {

namespace {

// Block-wide max reduction. All NUM_THREADS threads must participate.
template <int NUM_THREADS>
__device__ __forceinline__ float blockReduceMax(float value) {
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    __shared__ float warp_max[NUM_WARPS];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    if (lane == 0) {
        warp_max[warp] = value;
    }
    __syncthreads();

    float result = (threadIdx.x < NUM_WARPS) ? warp_max[threadIdx.x] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            result = fmaxf(result, __shfl_xor_sync(0xffffffff, result, offset));
        }
        if (lane == 0) {
            warp_max[0] = result;
        }
    }
    __syncthreads();
    result = warp_max[0];
    __syncthreads();
    return result;
}

// Quantize one head_dim vector (src -> dst) and write its dequant scale.
// All NUM_THREADS threads of the block must call this together.
template <typename Tdata, int NUM_THREADS>
__device__ __forceinline__ void quantizeHeadVector(
    uint8_t *dst,
    float *scale_out,
    const Tdata *src,
    const size_t num_dims) {
    float amax = 0.0f;
    for (int i = threadIdx.x; i < num_dims; i += NUM_THREADS) {
        amax = fmaxf(amax, fabsf(static_cast<float>(src[i])));
    }
    amax = blockReduceMax<NUM_THREADS>(amax);

    const float scale = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
    if (threadIdx.x == 0) {
        *scale_out = scale;
    }
    const float inv_scale = 1.0f / scale;
    for (int i = threadIdx.x; i < num_dims; i += NUM_THREADS) {
        dst[i] = infiniopFp8E4m3Encode(static_cast<float>(src[i]) * inv_scale);
    }
    // Keep the whole block in lockstep for the next vector's reduction.
    __syncthreads();
}

} // namespace

template <
    typename Tdata, // Data type of the source K/V tensors (half, __nv_bfloat16)
    int NUM_THREADS // Number of threads per block, configured at launch time
    >
__device__ void pagedCachingFp8Kernel(
    // ----- Output Tensors (F8 codes stored as raw bytes) -----
    uint8_t *k_cache_ptr, // [num_blocks, nkvh, block_size, dh]
    uint8_t *v_cache_ptr, // [num_blocks, nkvh, block_size, dv]
    float *k_scale_ptr,   // [num_blocks, nkvh, block_size]
    float *v_scale_ptr,   // [num_blocks, nkvh, block_size]
    // ----- Input Tensors -----
    const Tdata *k_ptr,              // [ntok, nkvh, dh]
    const Tdata *v_ptr,              // [ntok, nkvh, dv]
    const int64_t *slot_mapping_ptr, // [ntok]
    // ----- Metadata -----
    const size_t head_size,   // Dimension of each key head (dh_k)
    const size_t v_head_size, // Dimension of each value head (dh_v)
    const size_t block_size,  // Number of tokens per block in the KV cache
    // ----- Stride Information (identical semantics to the copy kernel) -----
    const ptrdiff_t k_src_stride,
    const ptrdiff_t v_src_stride,
    const ptrdiff_t k_src_head_stride,
    const ptrdiff_t v_src_head_stride,
    const ptrdiff_t k_cache_block_stride,
    const ptrdiff_t v_cache_block_stride,
    const ptrdiff_t k_cache_head_stride,
    const ptrdiff_t v_cache_head_stride,
    const ptrdiff_t k_cache_slot_stride,
    const ptrdiff_t v_cache_slot_stride,
    // ----- Scale strides ([num_blocks, nkvh, block_size]) -----
    const ptrdiff_t k_scale_block_stride,
    const ptrdiff_t k_scale_head_stride,
    const ptrdiff_t k_scale_slot_stride,
    const ptrdiff_t v_scale_block_stride,
    const ptrdiff_t v_scale_head_stride,
    const ptrdiff_t v_scale_slot_stride) {

    const int token_idx = blockIdx.y;
    const int head_idx = blockIdx.x;

    const int64_t slot_idx = slot_mapping_ptr[token_idx];
    if (slot_idx < 0) {
        return;
    }
    const int64_t physical_block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    const Tdata *k_src_head_ptr = k_ptr + token_idx * k_src_stride + head_idx * k_src_head_stride;
    const Tdata *v_src_head_ptr = v_ptr + token_idx * v_src_stride + head_idx * v_src_head_stride;

    uint8_t *k_dst_head_ptr = k_cache_ptr + physical_block_idx * k_cache_block_stride + head_idx * k_cache_head_stride + block_offset * k_cache_slot_stride;
    uint8_t *v_dst_head_ptr = v_cache_ptr + physical_block_idx * v_cache_block_stride + head_idx * v_cache_head_stride + block_offset * v_cache_slot_stride;

    float *k_scale_out = k_scale_ptr + physical_block_idx * k_scale_block_stride + head_idx * k_scale_head_stride + block_offset * k_scale_slot_stride;
    float *v_scale_out = v_scale_ptr + physical_block_idx * v_scale_block_stride + head_idx * v_scale_head_stride + block_offset * v_scale_slot_stride;

    quantizeHeadVector<Tdata, NUM_THREADS>(k_dst_head_ptr, k_scale_out, k_src_head_ptr, head_size);
    quantizeHeadVector<Tdata, NUM_THREADS>(v_dst_head_ptr, v_scale_out, v_src_head_ptr, v_head_size);
}

} // namespace op::paged_caching::cuda

#endif // __PAGED_CACHING_FP8_KERNEL_CUH__
