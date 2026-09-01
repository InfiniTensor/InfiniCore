#include <cuda_runtime.h>

#include <cstdint>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel_fp8.cuh"

// FP8(E4M3) KV-cache decode launchers. Independent of the F16/BF16 kernel
// family: k_cache/v_cache carry E4M3 codes and k_scale/v_scale the per-token
// F32 dequant scales produced by paged_caching. `dtype` is the q/out dtype
// (F16 or BF16); accumulation is always float.

namespace op::paged_attention::nvidia {

namespace {

template <typename Tindex, typename Tdata, int HEAD_SIZE>
INFINIOP_CUDA_KERNEL flashAttentionDecodeFp8(
    Tdata *out,
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
    ptrdiff_t v_scale_slot_stride) {
    op::paged_attention::cuda::flashAttentionDecodeFp8Kernel<Tindex, Tdata, HEAD_SIZE>(
        out, q, k_cache, v_cache, k_scale, v_scale, block_tables, cache_lens, alibi_slopes,
        num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,
        q_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        o_stride, o_head_stride,
        k_scale_block_stride, k_scale_head_stride, k_scale_slot_stride,
        v_scale_block_stride, v_scale_head_stride, v_scale_slot_stride);
}

template <typename Tindex>
infiniStatus_t launch_decode_fp8_impl(
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

    // One CTA per (sequence, query head). The block size follows the kernel's
    // warp count (kFp8DecodeNumWarps, default 8 warps = 256 threads); each
    // lane owns HEAD_SIZE/32 consecutive dims.
    const dim3 grid(static_cast<uint64_t>(num_heads), static_cast<uint64_t>(num_seqs), 1);
    constexpr uint32_t kBlockThreads = op::paged_attention::cuda::kFp8DecodeNumWarps * 32;

#define LAUNCH_FP8_DECODE(Tdata, HEAD_SIZE)                                                 \
    flashAttentionDecodeFp8<Tindex, Tdata, HEAD_SIZE>                                       \
        <<<grid, kBlockThreads, 0, stream>>>(                                                   \
            static_cast<Tdata *>(out),                                                      \
            static_cast<const Tdata *>(q),                                                  \
            static_cast<const uint8_t *>(k_cache),                                          \
            static_cast<const uint8_t *>(v_cache),                                          \
            static_cast<const float *>(k_scale),                                            \
            static_cast<const float *>(v_scale),                                            \
            block_tables, cache_lens, alibi_slopes,                                         \
            num_kv_heads, scale, max_num_blocks_per_seq, page_block_size,                   \
            q_stride, q_head_stride,                                                        \
            k_batch_stride, k_row_stride, k_head_stride,                                    \
            v_batch_stride, v_row_stride, v_head_stride,                                    \
            o_stride, o_head_stride,                                                        \
            k_scale_block_stride, k_scale_head_stride, k_scale_slot_stride,                 \
            v_scale_block_stride, v_scale_head_stride, v_scale_slot_stride)

#define DISPATCH_FP8_DECODE_HEAD_SIZE(HEAD_SIZE_)                 \
    do {                                                          \
        if (dtype == INFINI_DTYPE_F16) {                          \
            LAUNCH_FP8_DECODE(half, HEAD_SIZE_);                  \
            return INFINI_STATUS_SUCCESS;                         \
        }                                                         \
        if (dtype == INFINI_DTYPE_BF16) {                         \
            LAUNCH_FP8_DECODE(__nv_bfloat16, HEAD_SIZE_);         \
            return INFINI_STATUS_SUCCESS;                         \
        }                                                         \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                    \
    } while (false)

    switch (head_size) {
    case 64:
        DISPATCH_FP8_DECODE_HEAD_SIZE(64);
    case 128:
        DISPATCH_FP8_DECODE_HEAD_SIZE(128);
    default:
        // FP8 decode v1 implements head_size 64/128 only.
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }

#undef DISPATCH_FP8_DECODE_HEAD_SIZE
#undef LAUNCH_FP8_DECODE
}

} // namespace

#define DEFINE_LAUNCH_DECODE_FP8(SUFFIX, Tindex)                                   \
    infiniStatus_t launch_decode_fp8_##SUFFIX(                                     \
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
