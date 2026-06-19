#include <musa_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "../cuda/kernel_v2.cuh"

namespace op::paged_attention::moore {

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd192Warp(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
    op::paged_attention::cuda::flashAttentionDecodeWarpKernel<Tindex, Tdata, 192>(
        out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, o_stride);
}

template <typename Tindex, typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd192SplitKv(
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    const Tdata *q,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    int num_splits) {
    op::paged_attention::cuda::flashAttentionDecodeSplitKvWarpKernel<Tindex, Tdata, 192>(
        partial_acc, partial_m, partial_l,
        q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size, q_stride,
        k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
        v_head_stride, num_splits);
}

template <typename Tdata>
INFINIOP_MOORE_KERNEL flashAttentionDecodeHd192SplitKvCombine(
    Tdata *out,
    const float *partial_acc,
    const float *partial_m,
    const float *partial_l,
    int num_splits,
    ptrdiff_t o_stride) {
    op::paged_attention::cuda::flashAttentionDecodeSplitKvCombineWarpKernel<Tdata, 192>(
        out, partial_acc, partial_m, partial_l, num_splits, o_stride);
}

template <typename Tindex>
infiniStatus_t launch_decode_hd192_impl(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    infiniDtype_t dtype,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    musaStream_t stream) {
    (void)workspace;
    (void)workspace_size;

    const dim3 grid(static_cast<uint64_t>(num_heads), static_cast<uint64_t>(num_seqs), 1);
    const dim3 block(32);

    constexpr int kMaxSplits = 8;
    bool use_split = true;
    if (const char *env = std::getenv("INFINIOP_FLASH_DECODE_SPLITKV")) {
        use_split = !(std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0);
    }
    int num_splits = 4;
    if (const char *env = std::getenv("INFINIOP_FLASH_NUM_SPLITS")) {
        const int v = std::atoi(env);
        if (v > 0) {
            num_splits = v;
        }
    }
    if (num_splits < 1) {
        num_splits = 1;
    }
    if (num_splits > kMaxSplits) {
        num_splits = kMaxSplits;
    }
    use_split = use_split && num_splits > 1;

    if (use_split) {
        const size_t n = num_seqs * num_heads;
        const size_t acc_elems = static_cast<size_t>(num_splits) * n * 192;
        const size_t m_elems = static_cast<size_t>(num_splits) * n;
        const size_t l_elems = static_cast<size_t>(num_splits) * n;
        const size_t needed_bytes = (acc_elems + m_elems + l_elems) * sizeof(float);
        if (workspace == nullptr || workspace_size < needed_bytes) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        float *ws = static_cast<float *>(workspace);
        float *partial_acc = ws;
        float *partial_m = partial_acc + acc_elems;
        float *partial_l = partial_m + m_elems;
        const dim3 grid_split(static_cast<uint64_t>(num_heads), static_cast<uint64_t>(num_seqs), static_cast<uint64_t>(num_splits));

        if (dtype == INFINI_DTYPE_F16) {
            flashAttentionDecodeHd192SplitKv<Tindex, half><<<grid_split, block, 0, stream>>>(
                partial_acc, partial_m, partial_l,
                static_cast<const half *>(q),
                static_cast<const half *>(k_cache),
                static_cast<const half *>(v_cache),
                block_tables, cache_lens, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                q_stride, k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride, num_splits);
            flashAttentionDecodeHd192SplitKvCombine<half><<<grid, block, 0, stream>>>(
                static_cast<half *>(out), partial_acc, partial_m, partial_l, num_splits, o_stride);
            return INFINI_STATUS_SUCCESS;
        }
        if (dtype == INFINI_DTYPE_BF16) {
            flashAttentionDecodeHd192SplitKv<Tindex, __nv_bfloat16><<<grid_split, block, 0, stream>>>(
                partial_acc, partial_m, partial_l,
                static_cast<const __nv_bfloat16 *>(q),
                static_cast<const __nv_bfloat16 *>(k_cache),
                static_cast<const __nv_bfloat16 *>(v_cache),
                block_tables, cache_lens, alibi_slopes,
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
                q_stride, k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride, num_splits);
            flashAttentionDecodeHd192SplitKvCombine<__nv_bfloat16><<<grid, block, 0, stream>>>(
                static_cast<__nv_bfloat16 *>(out), partial_acc, partial_m, partial_l, num_splits, o_stride);
            return INFINI_STATUS_SUCCESS;
        }
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (dtype == INFINI_DTYPE_F16) {
        flashAttentionDecodeHd192Warp<Tindex, half><<<grid, block, 0, stream>>>(
            static_cast<half *>(out),
            static_cast<const half *>(q),
            static_cast<const half *>(k_cache),
            static_cast<const half *>(v_cache),
            block_tables, cache_lens, alibi_slopes,
            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
            q_stride, k_batch_stride, k_row_stride, k_head_stride,
            v_batch_stride, v_row_stride, v_head_stride, o_stride);
        return INFINI_STATUS_SUCCESS;
    }
    if (dtype == INFINI_DTYPE_BF16) {
        flashAttentionDecodeHd192Warp<Tindex, __nv_bfloat16><<<grid, block, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(out),
            static_cast<const __nv_bfloat16 *>(q),
            static_cast<const __nv_bfloat16 *>(k_cache),
            static_cast<const __nv_bfloat16 *>(v_cache),
            block_tables, cache_lens, alibi_slopes,
            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
            q_stride, k_batch_stride, k_row_stride, k_head_stride,
            v_batch_stride, v_row_stride, v_head_stride, o_stride);
        return INFINI_STATUS_SUCCESS;
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

infiniStatus_t launch_decode_hd192_i64(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const int64_t *block_tables, const int64_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    musaStream_t stream) {
    return launch_decode_hd192_impl<int64_t>(
        workspace, workspace_size, out, q, k_cache, v_cache, dtype, block_tables, cache_lens, alibi_slopes,
        num_heads, num_seqs, num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        q_stride, k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride, o_stride, stream);
}

infiniStatus_t launch_decode_hd192_i32(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const int32_t *block_tables, const int32_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    musaStream_t stream) {
    return launch_decode_hd192_impl<int32_t>(
        workspace, workspace_size, out, q, k_cache, v_cache, dtype, block_tables, cache_lens, alibi_slopes,
        num_heads, num_seqs, num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        q_stride, k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride, o_stride, stream);
}

infiniStatus_t launch_decode_hd192_u32(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const uint32_t *block_tables, const uint32_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    musaStream_t stream) {
    return launch_decode_hd192_impl<uint32_t>(
        workspace, workspace_size, out, q, k_cache, v_cache, dtype, block_tables, cache_lens, alibi_slopes,
        num_heads, num_seqs, num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        q_stride, k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride, o_stride, stream);
}

} // namespace op::paged_attention::moore
