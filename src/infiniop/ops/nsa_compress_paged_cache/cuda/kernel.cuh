#ifndef __NSA_COMPRESS_PAGED_CACHE_CUDA_KERNEL_CUH__
#define __NSA_COMPRESS_PAGED_CACHE_CUDA_KERNEL_CUH__

#include <cstddef>
#include <cstdint>
// On MetaX these headers do not exist; the equivalents come from
// devices/metax/metax_kernel_common.h, which the .maca translation unit
// includes before this one.
#if !defined(__MACA__) && !defined(__MACACC__)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace op::nsa_compress_paged_cache::cuda {

template <typename T>
__device__ inline float loadFloat(const T *ptr) { return static_cast<float>(*ptr); }
template <>
__device__ inline float loadFloat<__half>(const __half *ptr) { return __half2float(*ptr); }
template <>
__device__ inline float loadFloat<__nv_bfloat16>(const __nv_bfloat16 *ptr) { return __bfloat162float(*ptr); }
template <typename T>
__device__ inline void storeFloat(T *ptr, float value) { *ptr = static_cast<T>(value); }
template <>
__device__ inline void storeFloat<__half>(__half *ptr, float value) { *ptr = __float2half(value); }
template <>
__device__ inline void storeFloat<__nv_bfloat16>(__nv_bfloat16 *ptr, float value) { *ptr = __float2bfloat16(value); }

template <typename Tindex, typename Tdata>
__device__ void compressPagedCacheKernel(
    Tdata *k_cmp,
    Tdata *v_cmp,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    size_t subblocks_per_page,
    int nsa_block_size,
    int update_last_only,
    ptrdiff_t k_cmp_block_stride,
    ptrdiff_t k_cmp_head_stride,
    ptrdiff_t v_cmp_block_stride,
    ptrdiff_t v_cmp_head_stride,
    ptrdiff_t k_cache_batch_stride,
    ptrdiff_t k_cache_head_stride,
    ptrdiff_t k_cache_row_stride,
    ptrdiff_t v_cache_batch_stride,
    ptrdiff_t v_cache_head_stride,
    ptrdiff_t v_cache_row_stride,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t cache_lens_stride) {
    const size_t seq = blockIdx.x;
    size_t logical_nsa_block = blockIdx.y;
    const size_t kv_head = blockIdx.z;
    const int dim = threadIdx.x;
    constexpr int kHeadDim = 128;
    if (dim >= kHeadDim) {
        return;
    }

    const int64_t seq_len = static_cast<int64_t>(cache_lens[seq * cache_lens_stride]);
    if (seq_len <= 0) {
        return;
    }
    if (update_last_only) {
        logical_nsa_block = static_cast<size_t>((seq_len - 1) / nsa_block_size);
    }
    const int64_t tok_begin = static_cast<int64_t>(logical_nsa_block) * nsa_block_size;
    if (tok_begin >= seq_len) {
        return;
    }
    const int64_t tok_end = (tok_begin + nsa_block_size < seq_len) ? (tok_begin + nsa_block_size) : seq_len;
    const size_t logical_page = static_cast<size_t>(tok_begin / static_cast<int64_t>(page_block_size));
    if (logical_page >= max_num_blocks_per_seq) {
        return;
    }
    const size_t subblock = static_cast<size_t>((tok_begin % static_cast<int64_t>(page_block_size)) / nsa_block_size);
    const Tindex physical = block_tables[seq * block_table_batch_stride + logical_page];
    const size_t cmp_block = static_cast<size_t>(physical) * subblocks_per_page + subblock;
    float k_sum = 0.0f;
    float v_sum = 0.0f;
    for (int64_t tok = tok_begin; tok < tok_end; ++tok) {
        const size_t row = static_cast<size_t>(tok % static_cast<int64_t>(page_block_size));
        const size_t k_base = static_cast<size_t>(physical) * k_cache_batch_stride + kv_head * k_cache_head_stride + row * k_cache_row_stride;
        const size_t v_base = static_cast<size_t>(physical) * v_cache_batch_stride + kv_head * v_cache_head_stride + row * v_cache_row_stride;
        k_sum += loadFloat(k_cache + k_base + dim);
        v_sum += loadFloat(v_cache + v_base + dim);
    }
    const float inv = 1.0f / static_cast<float>(tok_end - tok_begin);
    storeFloat(k_cmp + cmp_block * k_cmp_block_stride + kv_head * k_cmp_head_stride + dim, k_sum * inv);
    storeFloat(v_cmp + cmp_block * v_cmp_block_stride + kv_head * v_cmp_head_stride + dim, v_sum * inv);
}

} // namespace op::nsa_compress_paged_cache::cuda

#endif // __NSA_COMPRESS_PAGED_CACHE_CUDA_KERNEL_CUH__
