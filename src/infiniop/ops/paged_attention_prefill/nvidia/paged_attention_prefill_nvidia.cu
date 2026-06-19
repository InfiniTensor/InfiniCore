#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ALI_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

// #include "paged_attention_prefill_fa2.cuh"
#include "paged_attention_prefill_nvidia.cuh"

#include "../cuda/kernel_v2.cuh"

namespace op::paged_attention_prefill::nvidia {

namespace {
constexpr size_t ceilDiv(size_t a, size_t b) {
    return (a + b - 1) / b;
}

inline const char *default_prefill_kernel(const PagedAttentionPrefillInfo &info) {
    if (info.head_size == 576) {
        return "ref";
    }
    // Iluvatar/Hygon: use warp for the non-MLA shapes where it is the stable path.
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    (void)info;
    return "warp";
#endif
    if (info.head_size == 192) {
        return "warp";
    }
    // Heuristic auto-dispatch (v0.4):
    // - Prefer the pipelined + tile-wise softmax kernel on FA2-compatible block_size=256.
    // - Keep a conservative fallback for other shapes / older GPUs (cp.async is a no-op below SM80).
    //
    // Users can always override via INFINIOP_FLASH_PREFILL_KERNEL.
    if (info.page_block_size == 256 && (info.dtype == INFINI_DTYPE_F16 || info.dtype == INFINI_DTYPE_BF16)) {
        if (info.head_size == 128) {
            return "warpcta8pipe";
        }
        // For head_size=64 we keep the previous default until we have broader perf coverage.
    }
    return "warpcta8";
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128Warp(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // Legacy per-seq launch (kept only as a wrapper; current "warp" impl uses a global-token kernel).
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpKernel<Tindex, Tdata, 128>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, block_table_batch_stride,
        q_stride, q_head_stride, k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride, o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd64Warp(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // Legacy per-seq launch (kept only as a wrapper; current "warp" impl uses a global-token kernel).
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpKernel<Tindex, Tdata, 64>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, block_table_batch_stride,
        q_stride, q_head_stride, k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride, o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128WarpCta(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 4 warps per CTA, one warp per query token.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernel<Tindex, Tdata, 128, 4, 64>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd64WarpCta(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 4 warps per CTA, one warp per query token.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernel<Tindex, Tdata, 64, 4, 128>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128WarpCta8(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 8 warps per CTA, one warp per query token.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernel<Tindex, Tdata, 128, 8, 64>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128WarpCta8N128(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 8 warps per CTA, one warp per query token, tile_n=128 for fewer K stages.
    // Note: we keep K in shared memory but load V from global to stay within the per-block shared limit.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernelKOnly<Tindex, Tdata, 128, 8, 128>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd64WarpCta8(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 8 warps per CTA, one warp per query token.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernel<Tindex, Tdata, 64, 8, 128>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128WarpCta8Pipe(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 8 warps per CTA, one warp per query token, with cp.async pipelining.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernelPipelined<Tindex, Tdata, 128, 8, 32, 2>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

#if !defined(ENABLE_HYGON_API)
template <typename Tindex>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128WarpCta8Mma(
    half *out,
    const half *q,
    const half *k_cache,
    const half *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCta8MmaHd128Kernel<Tindex>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}
#endif // !defined(ENABLE_HYGON_API)

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd64WarpCta8Pipe(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 8 warps per CTA, one warp per query token, with cp.async pipelining.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernelPipelined<Tindex, Tdata, 64, 8, 32, 2>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128WarpCta8PipeSplitKv(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    int num_splits,
    size_t total_q_tokens,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride) {
    // Encode (split_idx, m_block) into blockIdx.z to allow a single kernel launch:
    // blockIdx.z in [0, num_splits * num_m_blocks).
    const int num_m_blocks = static_cast<int>((total_q_tokens + 8 - 1) / 8);
    const int bz = static_cast<int>(blockIdx.z);
    const int split_idx = bz / num_m_blocks;
    const int m_block = bz - split_idx * num_m_blocks;
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernelPipelinedSplitKv<Tindex, Tdata, 128, 8, 32, 2>(
        partial_acc, partial_m, partial_l, split_idx, num_splits, m_block, total_q_tokens,
        q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd64WarpCta8PipeSplitKv(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    int num_splits,
    size_t total_q_tokens,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride) {
    const int num_m_blocks = static_cast<int>((total_q_tokens + 8 - 1) / 8);
    const int bz = static_cast<int>(blockIdx.z);
    const int split_idx = bz / num_m_blocks;
    const int m_block = bz - split_idx * num_m_blocks;
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernelPipelinedSplitKv<Tindex, Tdata, 64, 8, 32, 2>(
        partial_acc, partial_m, partial_l, split_idx, num_splits, m_block, total_q_tokens,
        q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride);
}

template <typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128SplitKvCombine(
    Tdata *out,
    const float *partial_acc,
    const float *partial_m,
    const float *partial_l,
    int num_splits,
    size_t total_q_tokens,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    op::paged_attention_prefill::cuda::PagedAttentionPrefillSplitKvCombineWarpKernel<Tdata, 128>(
        out, partial_acc, partial_m, partial_l, num_splits, total_q_tokens, o_stride, o_head_stride);
}

template <typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd64SplitKvCombine(
    Tdata *out,
    const float *partial_acc,
    const float *partial_m,
    const float *partial_l,
    int num_splits,
    size_t total_q_tokens,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    op::paged_attention_prefill::cuda::PagedAttentionPrefillSplitKvCombineWarpKernel<Tdata, 64>(
        out, partial_acc, partial_m, partial_l, num_splits, total_q_tokens, o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd128WarpCta16(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 16 warps per CTA, one warp per query token.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernel<Tindex, Tdata, 128, 16, 64>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_CUDA_KERNEL PagedAttentionPrefillHd64WarpCta16(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    // 16 warps per CTA, one warp per query token.
    op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpCtaKernel<Tindex, Tdata, 64, 16, 128>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride);
}

template <typename Tindex, typename Tdata, int QK_HEAD_SIZE, int VALUE_SIZE>
__global__ void PagedAttentionPrefillMlaWarpKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_heads,
    size_t num_kv_heads,
    float scale,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    size_t num_seqs) {

    constexpr int WARP_SIZE = 32;
    constexpr int Q_DIMS_PER_LANE = QK_HEAD_SIZE / WARP_SIZE;
    constexpr int V_DIMS_PER_LANE = VALUE_SIZE / WARP_SIZE;

    const size_t global_token_idx = static_cast<size_t>(blockIdx.x);
    const size_t head_idx = static_cast<size_t>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);

    if (head_idx >= num_heads || lane >= WARP_SIZE) {
        return;
    }

    const size_t seq_idx = op::paged_attention_prefill::cuda::find_seq_id<Tindex>(global_token_idx, cu_seqlens_q_, num_seqs);
    const size_t q_start = static_cast<size_t>(cu_seqlens_q_[seq_idx]);
    const size_t q_end = static_cast<size_t>(cu_seqlens_q_[seq_idx + 1]);
    const size_t q_token_idx = global_token_idx - q_start;
    const size_t q_len = q_end - q_start;
    const size_t total_kv_len = static_cast<size_t>(total_kv_lens_[seq_idx]);
    const size_t history_len = total_kv_len - q_len;
    const size_t causal_limit = history_len + q_token_idx;

    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;
    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float LOG2E = 1.4426950408889634f;
    const float scale_log2 = scale * LOG2E;

    const Tdata *q_vec = q_ + static_cast<int64_t>(global_token_idx) * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
    Tdata *out_vec = out_ + static_cast<int64_t>(global_token_idx) * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;
    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);
    const int pbs = static_cast<int>(page_block_size);

    float q_reg[Q_DIMS_PER_LANE];
    float acc[V_DIMS_PER_LANE];
#pragma unroll
    for (int i = 0; i < Q_DIMS_PER_LANE; ++i) {
        q_reg[i] = static_cast<float>(q_vec[lane * Q_DIMS_PER_LANE + i]);
    }
#pragma unroll
    for (int i = 0; i < V_DIMS_PER_LANE; ++i) {
        acc[i] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t page = t / static_cast<size_t>(pbs);
        const size_t off = t - page * static_cast<size_t>(pbs);
        const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
        const Tdata *k_vec = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;
        const Tdata *v_vec = v_cache_ + static_cast<int64_t>(phys) * v_batch_stride + static_cast<int64_t>(off) * v_row_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;

        float qk = 0.0f;
#pragma unroll
        for (int i = 0; i < Q_DIMS_PER_LANE; ++i) {
            const int dim = lane * Q_DIMS_PER_LANE + i;
            qk += q_reg[i] * static_cast<float>(k_vec[dim]);
        }
        qk = op::paged_attention::cuda::warpReduceSum(qk);

        float alpha = 1.0f;
        float beta = 0.0f;
        if (lane == 0) {
            float score = qk * scale_log2;
            if (alibi_slope != 0.0f) {
                score += (alibi_slope * static_cast<float>(t - causal_limit)) * LOG2E;
            }
            const float m_new = fmaxf(m, score);
            alpha = exp2f(m - m_new);
            beta = exp2f(score - m_new);
            l = l * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        beta = __shfl_sync(0xffffffff, beta, 0);

#pragma unroll
        for (int i = 0; i < V_DIMS_PER_LANE; ++i) {
            const int dim = lane * V_DIMS_PER_LANE + i;
            acc[i] = acc[i] * alpha + beta * static_cast<float>(v_vec[dim]);
        }
    }

    float inv_l = 0.0f;
    if (lane == 0) {
        inv_l = 1.0f / (l + 1e-6f);
    }
    inv_l = __shfl_sync(0xffffffff, inv_l, 0);

#pragma unroll
    for (int i = 0; i < V_DIMS_PER_LANE; ++i) {
        const int dim = lane * V_DIMS_PER_LANE + i;
        const float value = acc[i] * inv_l;
        if constexpr (std::is_same_v<Tdata, half>) {
            out_vec[dim] = __float2half_rn(value);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_vec[dim] = __float2bfloat16_rn(value);
        } else {
            out_vec[dim] = static_cast<Tdata>(value);
        }
    }
}

template <typename Tindex, typename Tdata, typename Tcompute, int QK_HEAD_SIZE, int VALUE_SIZE>
__global__ void PagedAttentionPrefillMlaReferenceKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_heads,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    size_t num_seqs) {

    const size_t global_token_idx = static_cast<size_t>(blockIdx.x);
    const size_t head_idx = static_cast<size_t>(blockIdx.y);
    const size_t dim_idx = static_cast<size_t>(threadIdx.x);

    if (dim_idx >= VALUE_SIZE || head_idx >= num_heads) {
        return;
    }

    const size_t seq_idx = op::paged_attention_prefill::cuda::find_seq_id<Tindex>(global_token_idx, cu_seqlens_q_, num_seqs);
    const size_t q_token_idx = global_token_idx - static_cast<size_t>(cu_seqlens_q_[seq_idx]);
    const size_t q_len = static_cast<size_t>(cu_seqlens_q_[seq_idx + 1] - cu_seqlens_q_[seq_idx]);

    const size_t total_kv_len = static_cast<size_t>(total_kv_lens_[seq_idx]);
    const size_t history_len = total_kv_len - q_len;
    const size_t causal_limit = history_len + q_token_idx;

    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;
    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];

    const Tdata *q_vec = q_ + static_cast<int64_t>(global_token_idx) * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
    Tdata *out_ptr = out_ + static_cast<int64_t>(global_token_idx) * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;
    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);
    const size_t pbs = page_block_size;

    Tcompute max_score = -INFINITY;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t page = t / pbs;
        const size_t off = t - page * pbs;
        const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
        const Tdata *k_vec = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;

        Tcompute score = 0;
        for (size_t d = 0; d < QK_HEAD_SIZE; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += static_cast<Tcompute>(alibi_slope * static_cast<float>(t - causal_limit));
        }
        if (score > max_score) {
            max_score = score;
        }
    }

    Tcompute sum_exp = 0;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t page = t / pbs;
        const size_t off = t - page * pbs;
        const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
        const Tdata *k_vec = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;

        Tcompute score = 0;
        for (size_t d = 0; d < QK_HEAD_SIZE; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += static_cast<Tcompute>(alibi_slope * static_cast<float>(t - causal_limit));
        }
        sum_exp += static_cast<Tcompute>(expf(static_cast<float>(score - max_score)));
    }

    const Tcompute inv_sum = static_cast<Tcompute>(1.0f) / (sum_exp + static_cast<Tcompute>(1e-6f));
    Tcompute acc = 0;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t page = t / pbs;
        const size_t off = t - page * pbs;
        const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
        const Tdata *k_vec = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;

        Tcompute score = 0;
        for (size_t d = 0; d < QK_HEAD_SIZE; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += static_cast<Tcompute>(alibi_slope * static_cast<float>(t - causal_limit));
        }
        const Tcompute prob = static_cast<Tcompute>(expf(static_cast<float>(score - max_score))) * inv_sum;
        const Tdata *v_vec = v_cache_ + static_cast<int64_t>(phys) * v_batch_stride + static_cast<int64_t>(off) * v_row_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;
        acc += prob * static_cast<Tcompute>(v_vec[dim_idx]);
    }

    out_ptr[dim_idx] = static_cast<Tdata>(acc);
}

template <typename Tindex, typename Tdata, typename Tcompute>
infiniStatus_t launch_prefill_ref(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    size_t value_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {

    const dim3 grid(static_cast<uint32_t>(total_q_tokens), static_cast<uint32_t>(num_heads), 1);
    const dim3 block(static_cast<uint32_t>(value_size), 1, 1);

    if (head_size == 576 && value_size == 512) {
        const dim3 mla_block(32, 1, 1);
        PagedAttentionPrefillMlaWarpKernel<Tindex, Tdata, 576, 512>
            <<<grid, mla_block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_kv_heads, scale, page_block_size,
                block_table_batch_stride, q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride, num_seqs);
        return INFINI_STATUS_SUCCESS;
    }

    if (head_size == 64) {
        op::paged_attention_prefill::cuda::PagedAttentionPrefillReferenceKernel<Tindex, Tdata, Tcompute, 64>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride, q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride, num_seqs);
        return INFINI_STATUS_SUCCESS;
    }

    if (head_size == 128) {
        op::paged_attention_prefill::cuda::PagedAttentionPrefillReferenceKernel<Tindex, Tdata, Tcompute, 128>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride, q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride, num_seqs);
        return INFINI_STATUS_SUCCESS;
    }

    if (head_size == 192) {
        op::paged_attention_prefill::cuda::PagedAttentionPrefillReferenceKernel<Tindex, Tdata, Tcompute, 192>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride, q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride, num_seqs);
        return INFINI_STATUS_SUCCESS;
    }

    if (head_size == 576) {
        op::paged_attention_prefill::cuda::PagedAttentionPrefillReferenceKernel<Tindex, Tdata, Tcompute, 576>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride, q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride, num_seqs);
        return INFINI_STATUS_SUCCESS;
    }

    return INFINI_STATUS_BAD_TENSOR_SHAPE;
}

template <typename Tindex, typename Tdata>
infiniStatus_t launch_prefill_warp(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {

    const dim3 block(32, 1, 1);
    // Global-token launch:
    // - dramatically reduces grid size vs the legacy (num_seqs * total_q_tokens) launch
    // - matches PagedAttention varlen (cu_seqlens) mental model better
    const dim3 grid(static_cast<uint32_t>(num_heads),
                    static_cast<uint32_t>(total_q_tokens),
                    1);

    switch (head_size) {
    case 64:
        op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpGlobalKernel<Tindex, Tdata, 64>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_seqs, num_kv_heads, total_q_tokens, scale, max_num_blocks_per_seq,
                page_block_size, block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    case 128:
        op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpGlobalKernel<Tindex, Tdata, 128>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_seqs, num_kv_heads, total_q_tokens, scale, max_num_blocks_per_seq,
                page_block_size, block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    case 192:
        op::paged_attention_prefill::cuda::PagedAttentionPrefillWarpGlobalKernel<Tindex, Tdata, 192>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_seqs, num_kv_heads, total_q_tokens, scale, max_num_blocks_per_seq,
                page_block_size, block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
}

template <typename Tindex, typename Tdata>
infiniStatus_t launch_prefill(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {

    constexpr int kWarps = 4;
    constexpr int kThreads = kWarps * 32;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<uint32_t>(num_heads),
                    static_cast<uint32_t>(num_seqs),
                    static_cast<uint32_t>(ceilDiv(total_q_tokens, static_cast<size_t>(kWarps))));

    switch (head_size) {
    case 64:
        PagedAttentionPrefillHd64WarpCta<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    case 128:
        PagedAttentionPrefillHd128WarpCta<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
}

template <typename Tindex, typename Tdata>
infiniStatus_t launch_prefill_warpcta8(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {

    constexpr int kWarps = 8;
    constexpr int kThreads = kWarps * 32;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<uint32_t>(num_heads),
                    static_cast<uint32_t>(num_seqs),
                    static_cast<uint32_t>(ceilDiv(total_q_tokens, static_cast<size_t>(kWarps))));

    switch (head_size) {
    case 64:
        PagedAttentionPrefillHd64WarpCta8<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    case 128:
        PagedAttentionPrefillHd128WarpCta8<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
}

template <typename Tindex, typename Tdata>
infiniStatus_t launch_prefill_warpcta8pipe(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {

    constexpr int kWarps = 8;
    constexpr int kThreads = kWarps * 32;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<uint32_t>(num_heads),
                    static_cast<uint32_t>(num_seqs),
                    static_cast<uint32_t>(ceilDiv(total_q_tokens, static_cast<size_t>(kWarps))));

    switch (head_size) {
    case 64:
        PagedAttentionPrefillHd64WarpCta8Pipe<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    case 128:
        PagedAttentionPrefillHd128WarpCta8Pipe<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
}

template <typename Tindex, typename Tdata>
infiniStatus_t launch_prefill_warpcta8mma(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {
#if defined(ENABLE_HYGON_API)
    return launch_prefill_warpcta8pipe<Tindex, Tdata>(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
        num_heads, num_seqs, num_kv_heads, total_q_tokens, head_size, scale,
        max_num_blocks_per_seq, page_block_size,
        block_table_batch_stride,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride, stream);
#else

    // Current WMMA kernel only supports fp16 + head_dim=128.
    if constexpr (!std::is_same_v<Tdata, half>) {
        return launch_prefill_warpcta8pipe<Tindex, Tdata>(
            out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
            num_heads, num_seqs, num_kv_heads, total_q_tokens, head_size, scale,
            max_num_blocks_per_seq, page_block_size,
            block_table_batch_stride,
            q_stride, q_head_stride,
            k_batch_stride, k_row_stride, k_head_stride,
            v_batch_stride, v_row_stride, v_head_stride,
            o_stride, o_head_stride, stream);
    }

    if (head_size != 128) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Guardrail: the current WMMA-score kernel is correctness-first and can be extremely slow on long prompts.
    // Allow power users to force it via INFINIOP_FLASH_PREFILL_MMA_FORCE=1.
    const char *force_env = std::getenv("INFINIOP_FLASH_PREFILL_MMA_FORCE");
    const bool force_mma = (force_env != nullptr) && (std::strcmp(force_env, "1") == 0);
    const size_t seqlen_k_est = max_num_blocks_per_seq * page_block_size;
    if (!force_mma && seqlen_k_est > 4096) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                         "[infiniop][paged_attention_prefill] warpcta8mma is experimental and very slow for long seqlen_k (est=%zu). "
                         "Falling back to warpcta8pipe. Set INFINIOP_FLASH_PREFILL_MMA_FORCE=1 to override.\n",
                         seqlen_k_est);
            warned = true;
        }
        return launch_prefill_warpcta8pipe<Tindex, Tdata>(
            out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
            num_heads, num_seqs, num_kv_heads, total_q_tokens, head_size, scale,
            max_num_blocks_per_seq, page_block_size,
            block_table_batch_stride,
            q_stride, q_head_stride,
            k_batch_stride, k_row_stride, k_head_stride,
            v_batch_stride, v_row_stride, v_head_stride,
            o_stride, o_head_stride, stream);
    }

    // WMMA requires SM70+. If not supported (or if we can't query), fall back to the pipelined SIMT kernel.
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        if (prop.major < 7) {
            return launch_prefill_warpcta8pipe<Tindex, Tdata>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_heads, num_seqs, num_kv_heads, total_q_tokens, head_size, scale,
                max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride, stream);
        }
    }

    constexpr int kWarps = 8;
    constexpr int kThreads = kWarps * 32;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<uint32_t>(num_heads),
                    static_cast<uint32_t>(num_seqs),
                    static_cast<uint32_t>(ceilDiv(total_q_tokens, static_cast<size_t>(16))));

    PagedAttentionPrefillHd128WarpCta8Mma<Tindex>
        <<<grid, block, 0, stream>>>(
            static_cast<half *>(out),
            static_cast<const half *>(q),
            static_cast<const half *>(k_cache),
            static_cast<const half *>(v_cache),
            block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
            block_table_batch_stride,
            q_stride, q_head_stride,
            k_batch_stride, k_row_stride, k_head_stride,
            v_batch_stride, v_row_stride, v_head_stride,
            o_stride, o_head_stride);
    return INFINI_STATUS_SUCCESS;
#endif // defined(ENABLE_HYGON_API)
}

template <typename Tindex, typename Tdata>
infiniStatus_t launch_prefill_warpcta8pipe_splitkv(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    int num_splits,
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {

    constexpr int kMaxSplits = 8;
    if (num_splits < 1) {
        num_splits = 1;
    }
    if (num_splits > kMaxSplits) {
        num_splits = kMaxSplits;
    }

    constexpr int kWarps = 8;
    constexpr int kThreads = kWarps * 32;
    const dim3 block(kThreads);
    const size_t num_m_blocks = ceilDiv(total_q_tokens, static_cast<size_t>(kWarps));
    // Single kernel launch with split_idx encoded in grid.z:
    // blockIdx.z in [0, num_splits * num_m_blocks).
    const dim3 grid(static_cast<uint32_t>(num_heads),
                    static_cast<uint32_t>(num_seqs),
                    static_cast<uint32_t>(num_m_blocks * static_cast<size_t>(num_splits)));

    switch (head_size) {
    case 64:
        PagedAttentionPrefillHd64WarpCta8PipeSplitKv<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                partial_acc, partial_m, partial_l, num_splits, total_q_tokens,
                q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride);
        break;
    case 128:
        PagedAttentionPrefillHd128WarpCta8PipeSplitKv<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                partial_acc, partial_m, partial_l, num_splits, total_q_tokens,
                q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Combine: one warp per (token, head).
    const dim3 block2(32);
    const dim3 grid2(static_cast<uint32_t>(num_heads), static_cast<uint32_t>(total_q_tokens), 1);
    switch (head_size) {
    case 64:
        PagedAttentionPrefillHd64SplitKvCombine<Tdata>
            <<<grid2, block2, 0, stream>>>(
                out, partial_acc, partial_m, partial_l, num_splits, total_q_tokens, o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    case 128:
        PagedAttentionPrefillHd128SplitKvCombine<Tdata>
            <<<grid2, block2, 0, stream>>>(
                out, partial_acc, partial_m, partial_l, num_splits, total_q_tokens, o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
}

template <typename Tindex, typename Tdata>
infiniStatus_t launch_prefill_warpcta8n128(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {

    constexpr int kWarps = 8;
    constexpr int kThreads = kWarps * 32;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<uint32_t>(num_heads),
                    static_cast<uint32_t>(num_seqs),
                    static_cast<uint32_t>(ceilDiv(total_q_tokens, static_cast<size_t>(kWarps))));

    // Only meaningful for head_dim=128.
    if (head_size != 128) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    PagedAttentionPrefillHd128WarpCta8N128<Tindex, Tdata>
        <<<grid, block, 0, stream>>>(
            out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
            block_table_batch_stride,
            q_stride, q_head_stride,
            k_batch_stride, k_row_stride, k_head_stride,
            v_batch_stride, v_row_stride, v_head_stride,
            o_stride, o_head_stride);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tindex, typename Tdata>
infiniStatus_t launch_prefill_warpcta16(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *total_kv_lens,
    const Tindex *cu_seqlens_q,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    cudaStream_t stream) {

    constexpr int kWarps = 16;
    constexpr int kThreads = kWarps * 32;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<uint32_t>(num_heads),
                    static_cast<uint32_t>(num_seqs),
                    static_cast<uint32_t>(ceilDiv(total_q_tokens, static_cast<size_t>(kWarps))));

    switch (head_size) {
    case 64:
        PagedAttentionPrefillHd64WarpCta16<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    case 128:
        PagedAttentionPrefillHd128WarpCta16<Tindex, Tdata>
            <<<grid, block, 0, stream>>>(
                out, q, k_cache, v_cache, block_tables, total_kv_lens, cu_seqlens_q, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                block_table_batch_stride,
                q_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_stride, o_head_stride);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
}
} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t total_kv_lens_desc,
    infiniopTensorDescriptor_t cum_seqlens_q_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {

    auto info = PagedAttentionPrefillInfo::create(
        out_desc, q_desc, k_cache_desc, v_cache_desc,
        block_tables_desc, total_kv_lens_desc, cum_seqlens_q_desc,
        alibi_slopes_desc, scale);
    CHECK_RESULT(info);

    // Optional split-kv prefill requires workspace for partial (m, l, acc).
    // IMPORTANT: Unlike decode, prefill's total_q_tokens can be very large, so we must NOT reserve
    // a huge workspace unless the user explicitly enables split-kv.
    bool use_splitkv = false;
    if (const char *env = std::getenv("INFINIOP_FLASH_PREFILL_SPLITKV")) {
        use_splitkv = (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
    }
    int num_splits = 1;
    if (use_splitkv) {
        if (const char *env = std::getenv("INFINIOP_FLASH_PREFILL_NUM_SPLITS")) {
            const int v = std::atoi(env);
            if (v > 0) {
                num_splits = v;
            }
        } else {
            num_splits = 4;
        }
        constexpr int kMaxSplits = 8;
        if (num_splits > kMaxSplits) {
            num_splits = kMaxSplits;
        }
    }
    const size_t n = info->total_q_tokens * info->num_heads;
    const size_t splitkv_workspace_bytes = use_splitkv ? (static_cast<size_t>(num_splits) * n * (info->head_size + 2) * sizeof(float)) : 0;

    const size_t workspace_bytes = splitkv_workspace_bytes;
    // const size_t workspace_bytes = splitkv_workspace_bytes + fa2_workspace_bytes;

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), workspace_bytes, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables,
    const void *total_kv_lens,
    const void *cum_seqlens_q,
    const void *alibi_slopes,
    void *stream_) const {
    auto stream = static_cast<cudaStream_t>(stream_);

    const float *alibi_ptr = (alibi_slopes == nullptr) ? nullptr : static_cast<const float *>(alibi_slopes);
    const void *total_kv_lens_ptr = total_kv_lens;
    const void *cu_seqlens_q_ptr = cum_seqlens_q;

    bool use_splitkv = false;
    if (const char *env = std::getenv("INFINIOP_FLASH_PREFILL_SPLITKV")) {
        use_splitkv = (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
    }
    int num_splits = 1;
    if (use_splitkv) {
        if (const char *env = std::getenv("INFINIOP_FLASH_PREFILL_NUM_SPLITS")) {
            const int v = std::atoi(env);
            if (v > 0) {
                num_splits = v;
            }
        } else {
            // Conservative default; users can override.
            num_splits = 4;
        }
        constexpr int kMaxSplits = 8;
        if (num_splits > kMaxSplits) {
            num_splits = kMaxSplits;
        }
        const size_t n = _info.total_q_tokens * _info.num_heads;
        const size_t required = static_cast<size_t>(num_splits) * n * (_info.head_size + 2) * sizeof(float);
        if (workspace_size < required) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
    }

    if (use_splitkv) {
        const size_t n = _info.total_q_tokens * _info.num_heads;
        float *partial_acc = static_cast<float *>(workspace);
        float *partial_m = partial_acc + static_cast<size_t>(num_splits) * n * _info.head_size;
        float *partial_l = partial_m + static_cast<size_t>(num_splits) * n;

        // Dispatch by (Tdata, Tindex). total_kv_lens + cu_seqlens_q are always int32,  but now we also support int64_t.
#define DISPATCH_SPLITKV(Tindex, Tdata, BT_PTR)                                            \
    return launch_prefill_warpcta8pipe_splitkv<Tindex, Tdata>(                             \
        partial_acc, partial_m, partial_l, num_splits,                                     \
        static_cast<Tdata *>(out),                                                         \
        static_cast<const Tdata *>(q),                                                     \
        static_cast<const Tdata *>(k_cache),                                               \
        static_cast<const Tdata *>(v_cache),                                               \
        static_cast<const Tindex *>(BT_PTR),                                               \
        static_cast<const Tindex *>(total_kv_lens_ptr),                                    \
        static_cast<const Tindex *>(cu_seqlens_q_ptr),                                     \
        alibi_ptr,                                                                         \
        _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,         \
        _info.head_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size, \
        _info.block_table_batch_stride,                                                    \
        _info.q_stride, _info.q_head_stride,                                               \
        _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                     \
        _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                     \
        _info.o_stride, _info.o_head_stride, stream)
        if (_info.dtype == INFINI_DTYPE_F16) {
            if (_info.index_dtype == INFINI_DTYPE_I64) {
                DISPATCH_SPLITKV(int64_t, half, block_tables);
            }
            if (_info.index_dtype == INFINI_DTYPE_I32) {
                DISPATCH_SPLITKV(int32_t, half, block_tables);
            }
            if (_info.index_dtype == INFINI_DTYPE_U32) {
                DISPATCH_SPLITKV(uint32_t, half, block_tables);
            }
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (_info.dtype == INFINI_DTYPE_BF16) {
            if (_info.index_dtype == INFINI_DTYPE_I64) {
                DISPATCH_SPLITKV(int64_t, __nv_bfloat16, block_tables);
            }
            if (_info.index_dtype == INFINI_DTYPE_I32) {
                DISPATCH_SPLITKV(int32_t, __nv_bfloat16, block_tables);
            }
            if (_info.index_dtype == INFINI_DTYPE_U32) {
                DISPATCH_SPLITKV(uint32_t, __nv_bfloat16, block_tables);
            }
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        return INFINI_STATUS_BAD_TENSOR_DTYPE;

#undef DISPATCH_SPLITKV
    }

// Default to the fastest validated kernel for supported shapes.
// "ref" is still available for debugging/correctness bisecting.
#define DISPATCH_KERNEL(Tindex, Tdata, Tcompute)                                                                                                                                                                                                                                                                                       \
    do {                                                                                                                                                                                                                                                                                                                               \
        const char *k_env = std::getenv("INFINIOP_FLASH_PREFILL_KERNEL");                                                                                                                                                                                                                                                              \
        const char *k = (k_env == nullptr) ? default_prefill_kernel(_info) : k_env;                                                                                                                                                                                                                                                    \
        if (k_env != nullptr) {                                                                                                                                                                                                                                                                                                        \
            const bool known = (std::strcmp(k, "warp") == 0) || (std::strcmp(k, "warpcta") == 0) || (std::strcmp(k, "warpcta8") == 0) || (std::strcmp(k, "warpcta8pipe") == 0) || (std::strcmp(k, "warpcta8mma") == 0) || (std::strcmp(k, "warpcta8n128") == 0) || (std::strcmp(k, "warpcta16") == 0) || (std::strcmp(k, "ref") == 0); \
            if (!known) {                                                                                                                                                                                                                                                                                                              \
                const char *fallback = default_prefill_kernel(_info);                                                                                                                                                                                                                                                                  \
                std::fprintf(stderr,                                                                                                                                                                                                                                                                                                   \
                             "[infiniop][paged_attention_prefill] WARNING: unknown kernel '%s', falling back to '%s'\n",                                                                                                                                                                                                               \
                             k, fallback);                                                                                                                                                                                                                                                                                             \
                k = fallback;                                                                                                                                                                                                                                                                                                          \
            }                                                                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        const char *dbg = std::getenv("INFINIOP_DEBUG_PREFILL_DISPATCH");                                                                                                                                                                                                                                                              \
        static bool printed_dispatch = false;                                                                                                                                                                                                                                                                                          \
        if (!printed_dispatch && dbg != nullptr && std::strcmp(dbg, "1") == 0) {                                                                                                                                                                                                                                                       \
            std::fprintf(stderr,                                                                                                                                                                                                                                                                                                       \
                         "[infiniop][paged_attention_prefill] kernel=%s (override=%s head_size=%zu block=%zu dtype=%zu)\n",                                                                                                                                                                                                            \
                         k,                                                                                                                                                                                                                                                                                                            \
                         (k_env == nullptr ? "auto" : "env"),                                                                                                                                                                                                                                                                          \
                         static_cast<size_t>(_info.head_size),                                                                                                                                                                                                                                                                         \
                         static_cast<size_t>(_info.page_block_size),                                                                                                                                                                                                                                                                   \
                         static_cast<size_t>(_info.dtype));                                                                                                                                                                                                                                                                            \
            printed_dispatch = true;                                                                                                                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                                                                                                                              \
        if (std::strcmp(k, "warp") == 0) {                                                                                                                                                                                                                                                                                             \
            return launch_prefill_warp<Tindex, Tdata>(                                                                                                                                                                                                                                                                                 \
                static_cast<Tdata *>(out), static_cast<const Tdata *>(q),                                                                                                                                                                                                                                                              \
                static_cast<const Tdata *>(k_cache), static_cast<const Tdata *>(v_cache),                                                                                                                                                                                                                                              \
                static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(total_kv_lens_ptr), static_cast<const Tindex *>(cu_seqlens_q_ptr), alibi_ptr,                                                                                                                                                                   \
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,                                                                                                                                                                                                                                             \
                _info.head_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size,                                                                                                                                                                                                                                     \
                _info.block_table_batch_stride,                                                                                                                                                                                                                                                                                        \
                _info.q_stride, _info.q_head_stride,                                                                                                                                                                                                                                                                                   \
                _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                                                                                                                                                                                                                                                         \
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                                                                                                                                                                                                                                                         \
                _info.o_stride, _info.o_head_stride, stream);                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        if (std::strcmp(k, "warpcta") == 0) {                                                                                                                                                                                                                                                                                          \
            return launch_prefill<Tindex, Tdata>(                                                                                                                                                                                                                                                                                      \
                static_cast<Tdata *>(out), static_cast<const Tdata *>(q),                                                                                                                                                                                                                                                              \
                static_cast<const Tdata *>(k_cache), static_cast<const Tdata *>(v_cache),                                                                                                                                                                                                                                              \
                static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(total_kv_lens_ptr), static_cast<const Tindex *>(cu_seqlens_q_ptr), alibi_ptr,                                                                                                                                                                   \
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,                                                                                                                                                                                                                                             \
                _info.head_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size,                                                                                                                                                                                                                                     \
                _info.block_table_batch_stride,                                                                                                                                                                                                                                                                                        \
                _info.q_stride, _info.q_head_stride,                                                                                                                                                                                                                                                                                   \
                _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                                                                                                                                                                                                                                                         \
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                                                                                                                                                                                                                                                         \
                _info.o_stride, _info.o_head_stride, stream);                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        if (std::strcmp(k, "warpcta8") == 0) {                                                                                                                                                                                                                                                                                         \
            return launch_prefill_warpcta8<Tindex, Tdata>(                                                                                                                                                                                                                                                                             \
                static_cast<Tdata *>(out), static_cast<const Tdata *>(q),                                                                                                                                                                                                                                                              \
                static_cast<const Tdata *>(k_cache), static_cast<const Tdata *>(v_cache),                                                                                                                                                                                                                                              \
                static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(total_kv_lens_ptr), static_cast<const Tindex *>(cu_seqlens_q_ptr), alibi_ptr,                                                                                                                                                                   \
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,                                                                                                                                                                                                                                             \
                _info.head_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size,                                                                                                                                                                                                                                     \
                _info.block_table_batch_stride,                                                                                                                                                                                                                                                                                        \
                _info.q_stride, _info.q_head_stride,                                                                                                                                                                                                                                                                                   \
                _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                                                                                                                                                                                                                                                         \
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                                                                                                                                                                                                                                                         \
                _info.o_stride, _info.o_head_stride, stream);                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        if (std::strcmp(k, "warpcta8pipe") == 0) {                                                                                                                                                                                                                                                                                     \
            return launch_prefill_warpcta8pipe<Tindex, Tdata>(                                                                                                                                                                                                                                                                         \
                static_cast<Tdata *>(out), static_cast<const Tdata *>(q),                                                                                                                                                                                                                                                              \
                static_cast<const Tdata *>(k_cache), static_cast<const Tdata *>(v_cache),                                                                                                                                                                                                                                              \
                static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(total_kv_lens_ptr), static_cast<const Tindex *>(cu_seqlens_q_ptr), alibi_ptr,                                                                                                                                                                   \
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,                                                                                                                                                                                                                                             \
                _info.head_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size,                                                                                                                                                                                                                                     \
                _info.block_table_batch_stride,                                                                                                                                                                                                                                                                                        \
                _info.q_stride, _info.q_head_stride,                                                                                                                                                                                                                                                                                   \
                _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                                                                                                                                                                                                                                                         \
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                                                                                                                                                                                                                                                         \
                _info.o_stride, _info.o_head_stride, stream);                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        if constexpr (std::is_same_v<Tdata, half>) {                                                                                                                                                                                                                                                                                   \
            if (std::strcmp(k, "warpcta8mma") == 0) {                                                                                                                                                                                                                                                                                  \
                return launch_prefill_warpcta8mma<Tindex, Tdata>(                                                                                                                                                                                                                                                                      \
                    static_cast<Tdata *>(out), static_cast<const Tdata *>(q),                                                                                                                                                                                                                                                          \
                    static_cast<const Tdata *>(k_cache), static_cast<const Tdata *>(v_cache),                                                                                                                                                                                                                                          \
                    static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(total_kv_lens_ptr), static_cast<const Tindex *>(cu_seqlens_q_ptr), alibi_ptr,                                                                                                                                                               \
                    _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,                                                                                                                                                                                                                                         \
                    _info.head_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size,                                                                                                                                                                                                                                 \
                    _info.block_table_batch_stride,                                                                                                                                                                                                                                                                                    \
                    _info.q_stride, _info.q_head_stride,                                                                                                                                                                                                                                                                               \
                    _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                                                                                                                                                                                                                                                     \
                    _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                                                                                                                                                                                                                                                     \
                    _info.o_stride, _info.o_head_stride, stream);                                                                                                                                                                                                                                                                      \
            }                                                                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        if (std::strcmp(k, "warpcta8n128") == 0) {                                                                                                                                                                                                                                                                                     \
            return launch_prefill_warpcta8n128<Tindex, Tdata>(                                                                                                                                                                                                                                                                         \
                static_cast<Tdata *>(out), static_cast<const Tdata *>(q),                                                                                                                                                                                                                                                              \
                static_cast<const Tdata *>(k_cache), static_cast<const Tdata *>(v_cache),                                                                                                                                                                                                                                              \
                static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(total_kv_lens_ptr), static_cast<const Tindex *>(cu_seqlens_q_ptr), alibi_ptr,                                                                                                                                                                   \
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,                                                                                                                                                                                                                                             \
                _info.head_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size,                                                                                                                                                                                                                                     \
                _info.block_table_batch_stride,                                                                                                                                                                                                                                                                                        \
                _info.q_stride, _info.q_head_stride,                                                                                                                                                                                                                                                                                   \
                _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                                                                                                                                                                                                                                                         \
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                                                                                                                                                                                                                                                         \
                _info.o_stride, _info.o_head_stride, stream);                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        if (std::strcmp(k, "warpcta16") == 0) {                                                                                                                                                                                                                                                                                        \
            return launch_prefill_warpcta16<Tindex, Tdata>(                                                                                                                                                                                                                                                                            \
                static_cast<Tdata *>(out), static_cast<const Tdata *>(q),                                                                                                                                                                                                                                                              \
                static_cast<const Tdata *>(k_cache), static_cast<const Tdata *>(v_cache),                                                                                                                                                                                                                                              \
                static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(total_kv_lens_ptr), static_cast<const Tindex *>(cu_seqlens_q_ptr), alibi_ptr,                                                                                                                                                                   \
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,                                                                                                                                                                                                                                             \
                _info.head_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size,                                                                                                                                                                                                                                     \
                _info.block_table_batch_stride,                                                                                                                                                                                                                                                                                        \
                _info.q_stride, _info.q_head_stride,                                                                                                                                                                                                                                                                                   \
                _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                                                                                                                                                                                                                                                         \
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                                                                                                                                                                                                                                                         \
                _info.o_stride, _info.o_head_stride, stream);                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        if (std::strcmp(k, "ref") == 0) {                                                                                                                                                                                                                                                                                              \
            return launch_prefill_ref<Tindex, Tdata, Tcompute>(                                                                                                                                                                                                                                                                        \
                static_cast<Tdata *>(out), static_cast<const Tdata *>(q),                                                                                                                                                                                                                                                              \
                static_cast<const Tdata *>(k_cache), static_cast<const Tdata *>(v_cache),                                                                                                                                                                                                                                              \
                static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(total_kv_lens_ptr), static_cast<const Tindex *>(cu_seqlens_q_ptr), alibi_ptr,                                                                                                                                                                   \
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.total_q_tokens,                                                                                                                                                                                                                                             \
                _info.head_size, _info.value_size, _info.scale, _info.max_num_blocks_per_seq, _info.page_block_size,                                                                                                                                                                                                                   \
                _info.block_table_batch_stride,                                                                                                                                                                                                                                                                                        \
                _info.q_stride, _info.q_head_stride,                                                                                                                                                                                                                                                                                   \
                _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,                                                                                                                                                                                                                                                         \
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                                                                                                                                                                                                                                                         \
                _info.o_stride, _info.o_head_stride, stream);                                                                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                                                              \
        return INFINI_STATUS_BAD_PARAM;                                                                                                                                                                                                                                                                                                \
    } while (false)

#define DISPATCH_INDEX(Tindex)                             \
    do {                                                   \
        if (_info.dtype == INFINI_DTYPE_F16) {             \
            DISPATCH_KERNEL(Tindex, half, float);          \
        }                                                  \
        if (_info.dtype == INFINI_DTYPE_BF16) {            \
            DISPATCH_KERNEL(Tindex, __nv_bfloat16, float); \
        }                                                  \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;             \
    } while (false)

    if (_info.index_dtype == INFINI_DTYPE_I64) {
        DISPATCH_INDEX(int64_t);
    } else if (_info.index_dtype == INFINI_DTYPE_I32) {
        DISPATCH_INDEX(int32_t);
    } else if (_info.index_dtype == INFINI_DTYPE_U32) {
        DISPATCH_INDEX(uint32_t);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::paged_attention_prefill::nvidia
#endif
