#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel_fp8.cuh"

// FP8(E4M3) KV-cache decode launchers. Independent of the F16/BF16 kernel
// family: k_cache/v_cache carry E4M3 codes and k_scale/v_scale the per-token
// F32 dequant scales produced by paged_caching. `dtype` is the q/out dtype
// (F16 or BF16); accumulation is always float.
//
// Split-kv (flash-decoding): when the waves heuristic (or an env override)
// picks num_splits > 1, the decode kernel runs with grid.z = num_splits and
// writes per-shard (m, l, acc) partials into the workspace, then a combine
// kernel merges them into `out`. The workspace layout matches the F16/BF16
// family: partial_acc [kFp8DecodeMaxSplits, num_seqs, num_heads, head_size]
// followed by partial_m / partial_l [kFp8DecodeMaxSplits, num_seqs, num_heads].

namespace op::paged_attention::nvidia {

namespace {

constexpr size_t ceilDiv(size_t a, size_t b) {
    return (a + b - 1) / b;
}

inline int getSmCount() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 0;
    }
    int sm_count = 0;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return 0;
    }
    return sm_count;
}

// Same FA2-style "waves" heuristic as the F16/BF16 split-kv launchers: shard
// the KV sequence so that base_blocks * num_splits CTAs fill the SMs without
// paying for more combine work than the split saves. seqlen_k is an upper
// bound (max pages * page size).
inline int chooseNumSplitsHeuristic(size_t num_heads, size_t num_seqs, size_t seqlen_k, int sm_count) {
    if (sm_count <= 0) {
        return 1;
    }
    if (num_heads == 0 || num_seqs == 0) {
        return 1;
    }
    if (seqlen_k <= 256) {
        return 1;
    }

    const size_t base_blocks = num_heads * num_seqs;
    int best_splits = 1;
    // Baseline: one kernel, base_blocks CTAs, each scanning seqlen_k tokens.
    size_t best_score = (ceilDiv(base_blocks, static_cast<size_t>(sm_count)) * seqlen_k);

    size_t prev_work_per_block = seqlen_k;
    for (int s = 2; s <= op::paged_attention::cuda::kFp8DecodeMaxSplits; ++s) {
        const size_t blocks = base_blocks * static_cast<size_t>(s);
        const size_t waves_split = ceilDiv(blocks, static_cast<size_t>(sm_count));
        const size_t work_per_block = ceilDiv(seqlen_k, static_cast<size_t>(s));
        // If this split count doesn't reduce per-block work vs the previous split, it's effectively redundant.
        if (work_per_block == prev_work_per_block) {
            continue;
        }
        prev_work_per_block = work_per_block;
        // Combine is one extra kernel with base_blocks blocks; approximate as one more wave unit.
        const size_t waves_combine = ceilDiv(base_blocks, static_cast<size_t>(sm_count));
        const size_t score = waves_split * work_per_block + waves_combine;
        if (score < best_score) {
            best_score = score;
            best_splits = s;
        }
    }
    return best_splits;
}

template <typename Tindex, typename Tdata, int HEAD_SIZE>
INFINIOP_CUDA_KERNEL flashAttentionDecodeFp8(
    Tdata *out,
    float *partial_acc,
    float *partial_m,
    float *partial_l,
    const Tdata *q,
    const uint8_t *k_cache,
    const uint8_t *v_cache,
    const float *k_scale,
    const float *v_scale,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
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
    ptrdiff_t k_scale_block_stride,
    ptrdiff_t k_scale_head_stride,
    ptrdiff_t k_scale_slot_stride,
    ptrdiff_t v_scale_block_stride,
    ptrdiff_t v_scale_head_stride,
    ptrdiff_t v_scale_slot_stride,
    int num_splits) {
    op::paged_attention::cuda::flashAttentionDecodeFp8Kernel<Tindex, Tdata, HEAD_SIZE>(
        out, partial_acc, partial_m, partial_l,
        q, k_cache, v_cache, k_scale, v_scale, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride,
        k_scale_block_stride, k_scale_head_stride, k_scale_slot_stride,
        v_scale_block_stride, v_scale_head_stride, v_scale_slot_stride,
        num_splits);
}

template <typename Tdata, int HEAD_SIZE>
INFINIOP_CUDA_KERNEL flashAttentionDecodeFp8SplitKvCombine(
    Tdata *out,
    const float *partial_acc,
    const float *partial_m,
    const float *partial_l,
    int num_splits,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
    op::paged_attention::cuda::flashAttentionDecodeFp8SplitKvCombineKernel<Tdata, HEAD_SIZE>(
        out, partial_acc, partial_m, partial_l, num_splits, o_stride, o_head_stride);
}

// Split-kv policy for the FP8 decode kernel. Default (no env) is "auto": the
// waves heuristic splits only when it profits (small grids, long contexts).
// Env knobs mirror the F16/BF16 family:
//   INFINIOP_FLASH_DECODE_SPLITKV = 0/false (never) | 1/true (force) | auto
//   INFINIOP_FLASH_NUM_SPLITS     = 1..8 (fixed, implies split) | auto
inline int chooseNumSplitsFp8(size_t num_heads, size_t num_seqs,
                              size_t max_num_blocks_per_seq, size_t page_block_size) {
    const char *splitkv = std::getenv("INFINIOP_FLASH_DECODE_SPLITKV");
    if (splitkv && (std::strcmp(splitkv, "0") == 0 || std::strcmp(splitkv, "false") == 0)) {
        return 1;
    }
    const bool forced = splitkv && (std::strcmp(splitkv, "1") == 0 || std::strcmp(splitkv, "true") == 0);

    int num_splits = 1;
    const char *ns = std::getenv("INFINIOP_FLASH_NUM_SPLITS");
    const bool ns_fixed = ns && std::strcmp(ns, "auto") != 0 && std::atoi(ns) > 0;
    if (ns_fixed) {
        num_splits = std::atoi(ns);
    } else if (forced) {
        num_splits = 4; // fixed default, matching the F16/BF16 hd128 launcher
    } else {
        const size_t seqlen_k = max_num_blocks_per_seq * page_block_size;
        num_splits = chooseNumSplitsHeuristic(num_heads, num_seqs, seqlen_k, getSmCount());
    }
    if (num_splits < 1) {
        num_splits = 1;
    }
    if (num_splits > op::paged_attention::cuda::kFp8DecodeMaxSplits) {
        num_splits = op::paged_attention::cuda::kFp8DecodeMaxSplits;
    }

    if (const char *dbg = std::getenv("INFINIOP_FLASH_DEBUG_SPLITS")) {
        if (std::strcmp(dbg, "1") == 0 || std::strcmp(dbg, "true") == 0) {
            static size_t last_seqs = ~static_cast<size_t>(0);
            static size_t last_heads = ~static_cast<size_t>(0);
            static size_t last_cap = ~static_cast<size_t>(0);
            static int last_splits = -1;
            const size_t cap = max_num_blocks_per_seq * page_block_size;
            if (num_seqs != last_seqs || num_heads != last_heads || cap != last_cap || num_splits != last_splits) {
                last_seqs = num_seqs;
                last_heads = num_heads;
                last_cap = cap;
                last_splits = num_splits;
                std::fprintf(stderr,
                             "[INFINIOP][paged_attention][fp8] splitkv: heads=%zu seqs=%zu seqlen_k~%zu -> num_splits=%d\n",
                             num_heads, num_seqs, cap, num_splits);
            }
        }
    }
    return num_splits;
}

template <typename Tindex>
infiniStatus_t launch_decode_fp8_impl(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *k_scale,
    const void *v_scale,
    infiniDtype_t dtype,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const float *alibi_slopes,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    float scale,
    size_t head_size,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
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
    ptrdiff_t k_scale_block_stride,
    ptrdiff_t k_scale_head_stride,
    ptrdiff_t k_scale_slot_stride,
    ptrdiff_t v_scale_block_stride,
    ptrdiff_t v_scale_head_stride,
    ptrdiff_t v_scale_slot_stride,
    cudaStream_t stream) {

    if (num_heads == 0 || num_seqs == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    const int num_splits = chooseNumSplitsFp8(num_heads, num_seqs, max_num_blocks_per_seq, page_block_size);

    float *partial_acc = nullptr;
    float *partial_m = nullptr;
    float *partial_l = nullptr;
    if (num_splits > 1) {
        const size_t n = num_seqs * num_heads;
        const size_t acc_elems = static_cast<size_t>(op::paged_attention::cuda::kFp8DecodeMaxSplits) * n * head_size;
        const size_t ml_elems = static_cast<size_t>(op::paged_attention::cuda::kFp8DecodeMaxSplits) * n;
        const size_t needed_bytes = (acc_elems + 2 * ml_elems) * sizeof(float);
        if (workspace == nullptr || workspace_size < needed_bytes) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        float *ws = static_cast<float *>(workspace);
        partial_acc = ws;
        partial_m = partial_acc + acc_elems;
        partial_l = partial_m + ml_elems;
    }

    // One CTA per (sequence, query head, split shard). The block size follows
    // the kernel's warp count (kFp8DecodeNumWarps); each lane owns
    // HEAD_SIZE/32 consecutive dims. num_splits == 1 takes the same path with
    // grid.z == 1 and null partials, identical to the pre-split-kv kernel.
    const dim3 grid(static_cast<uint64_t>(num_heads), static_cast<uint64_t>(num_seqs),
                    static_cast<uint64_t>(num_splits));
    const dim3 grid_combine(static_cast<uint64_t>(num_heads), static_cast<uint64_t>(num_seqs), 1);
    constexpr uint32_t kBlockThreads = op::paged_attention::cuda::kFp8DecodeNumWarps * 32;

#define LAUNCH_FP8_DECODE(Tdata, HEAD_SIZE)                                                 \
    do {                                                                                    \
        flashAttentionDecodeFp8<Tindex, Tdata, HEAD_SIZE>                                   \
            <<<grid, kBlockThreads, 0, stream>>>(                                           \
                static_cast<Tdata *>(out),                                                  \
                partial_acc, partial_m, partial_l,                                          \
                static_cast<const Tdata *>(q),                                              \
                static_cast<const uint8_t *>(k_cache),                                      \
                static_cast<const uint8_t *>(v_cache),                                      \
                static_cast<const float *>(k_scale),                                        \
                static_cast<const float *>(v_scale),                                        \
                block_tables, cache_lens, alibi_slopes,                                     \
                num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,               \
                q_stride, q_head_stride,                                                    \
                k_batch_stride, k_row_stride, k_head_stride,                                \
                v_batch_stride, v_row_stride, v_head_stride,                                \
                o_stride, o_head_stride,                                                    \
                k_scale_block_stride, k_scale_head_stride, k_scale_slot_stride,             \
                v_scale_block_stride, v_scale_head_stride, v_scale_slot_stride,             \
                num_splits);                                                                \
        if (num_splits > 1) {                                                               \
            flashAttentionDecodeFp8SplitKvCombine<Tdata, HEAD_SIZE>                         \
                <<<grid_combine, HEAD_SIZE, 0, stream>>>(                                   \
                    static_cast<Tdata *>(out),                                              \
                    partial_acc, partial_m, partial_l,                                      \
                    num_splits, o_stride, o_head_stride);                                   \
        }                                                                                   \
        return INFINI_STATUS_SUCCESS;                                                       \
    } while (false)

#define DISPATCH_FP8_DECODE_HEAD_SIZE(HEAD_SIZE_)                 \
    do {                                                          \
        if (dtype == INFINI_DTYPE_F16) {                          \
            LAUNCH_FP8_DECODE(half, HEAD_SIZE_);                  \
        }                                                         \
        if (dtype == INFINI_DTYPE_BF16) {                         \
            LAUNCH_FP8_DECODE(__nv_bfloat16, HEAD_SIZE_);         \
        }                                                         \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                    \
    } while (false)

    switch (head_size) {
    case 64:
        DISPATCH_FP8_DECODE_HEAD_SIZE(64);
    case 128:
        DISPATCH_FP8_DECODE_HEAD_SIZE(128);
    default:
        // FP8 decode implements head_size 64/128 only.
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }

#undef DISPATCH_FP8_DECODE_HEAD_SIZE
#undef LAUNCH_FP8_DECODE
}

} // namespace

#define DEFINE_LAUNCH_DECODE_FP8(SUFFIX, Tindex)                                   \
    infiniStatus_t launch_decode_fp8_##SUFFIX(                                     \
        void *workspace, size_t workspace_size,                                    \
        void *out, const void *q, const void *k_cache, const void *v_cache,        \
        const void *k_scale, const void *v_scale,                                  \
        infiniDtype_t dtype,                                                       \
        const Tindex *block_tables, const Tindex *cache_lens,                      \
        const float *alibi_slopes,                                                 \
        size_t num_heads, size_t num_seqs, size_t num_kv_heads,                    \
        float scale, size_t head_size,                                             \
        size_t max_num_blocks_per_seq, size_t page_block_size,                     \
        ptrdiff_t q_stride, ptrdiff_t q_head_stride,                               \
        ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride, \
        ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, \
        ptrdiff_t o_stride, ptrdiff_t o_head_stride,                               \
        ptrdiff_t k_scale_block_stride, ptrdiff_t k_scale_head_stride,             \
        ptrdiff_t k_scale_slot_stride,                                             \
        ptrdiff_t v_scale_block_stride, ptrdiff_t v_scale_head_stride,             \
        ptrdiff_t v_scale_slot_stride,                                             \
        cudaStream_t stream) {                                                     \
        return launch_decode_fp8_impl<Tindex>(                                     \
            workspace, workspace_size,                                             \
            out, q, k_cache, v_cache, k_scale, v_scale, dtype,                     \
            block_tables, cache_lens, alibi_slopes,                                \
            num_heads, num_seqs, num_kv_heads, scale, head_size,                   \
            max_num_blocks_per_seq, page_block_size,                               \
            q_stride, q_head_stride,                                               \
            k_batch_stride, k_row_stride, k_head_stride,                           \
            v_batch_stride, v_row_stride, v_head_stride,                           \
            o_stride, o_head_stride,                                               \
            k_scale_block_stride, k_scale_head_stride, k_scale_slot_stride,        \
            v_scale_block_stride, v_scale_head_stride, v_scale_slot_stride,        \
            stream);                                                               \
    }

DEFINE_LAUNCH_DECODE_FP8(i64, int64_t)
DEFINE_LAUNCH_DECODE_FP8(i32, int32_t)
DEFINE_LAUNCH_DECODE_FP8(u32, uint32_t)

#undef DEFINE_LAUNCH_DECODE_FP8

} // namespace op::paged_attention::nvidia
