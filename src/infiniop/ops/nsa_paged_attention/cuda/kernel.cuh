#ifndef __NSA_PAGED_ATTENTION_CUDA_KERNEL_CUH__
#define __NSA_PAGED_ATTENTION_CUDA_KERNEL_CUH__

#include <cmath>
#include <cstddef>
#include <cstdint>
// On MetaX these headers do not exist; the equivalents come from
// devices/metax/metax_kernel_common.h, which the .maca translation unit
// includes before this one.
#if !defined(__MACA__) && !defined(__MACACC__)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace op::nsa_paged_attention::cuda {

template <typename T>
__device__ inline float loadFloat(const T *ptr) {
    return static_cast<float>(*ptr);
}

template <>
__device__ inline float loadFloat<__half>(const __half *ptr) {
    return __half2float(*ptr);
}

template <>
__device__ inline float loadFloat<__nv_bfloat16>(const __nv_bfloat16 *ptr) {
    return __bfloat162float(*ptr);
}

template <typename T>
__device__ inline void storeFloat(T *ptr, float value) {
    *ptr = static_cast<T>(value);
}

template <>
__device__ inline void storeFloat<__half>(__half *ptr, float value) {
    *ptr = __float2half(value);
}

template <>
__device__ inline void storeFloat<__nv_bfloat16>(__nv_bfloat16 *ptr, float value) {
    *ptr = __float2bfloat16(value);
}

__device__ inline float blockReduceSum128(float value) {
    constexpr unsigned kFullMask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API) && !defined(__MACA__) && !defined(__MACACC__)
#pragma unroll
#endif
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }

    __shared__ float warp_sums[4];
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    float sum = threadIdx.x < 4 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API) && !defined(__MACA__) && !defined(__MACACC__)
#pragma unroll
#endif
        for (int offset = 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(kFullMask, sum, offset);
        }
    }
    if (threadIdx.x == 0) {
        warp_sums[0] = sum;
    }
    __syncthreads();
    return warp_sums[0];
}

template <typename Tindex, typename Tdata, typename Tgate>
__device__ void nsaPagedDecodeHd128Kernel(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cmp,
    const Tdata *v_cmp,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const Tgate *gates,
    size_t num_heads,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    size_t subblocks_per_page,
    int nsa_block_size,
    int window_size,
    int select_blocks,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_cmp_block_stride,
    ptrdiff_t k_cmp_head_stride,
    ptrdiff_t v_cmp_block_stride,
    ptrdiff_t v_cmp_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t cache_lens_stride,
    ptrdiff_t gates_seq_stride,
    ptrdiff_t gates_branch_stride,
    ptrdiff_t gates_head_stride) {
    constexpr int kHeadDim = 128;

    const size_t seq = static_cast<size_t>(blockIdx.x);
    const size_t head = static_cast<size_t>(blockIdx.y);
    const int dim = threadIdx.x;
    if (dim >= kHeadDim) {
        return;
    }

    const size_t kv_head = head / (num_heads / num_kv_heads);
    const int64_t seq_len = static_cast<int64_t>(cache_lens[seq * cache_lens_stride]);
    const float qd = loadFloat(q + seq * q_stride + head * q_head_stride + dim);
    float comp_acc = 0.0f;
    float comp_m = -INFINITY;
    float comp_l = 0.0f;

    constexpr int kMaxSelectBlocks = 32;
    const int active_select_blocks = max(1, min(select_blocks, kMaxSelectBlocks));
    float top_scores[kMaxSelectBlocks];
    int top_blocks[kMaxSelectBlocks];
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API) && !defined(__MACA__) && !defined(__MACACC__)
#pragma unroll
#endif
    for (int i = 0; i < active_select_blocks; ++i) {
        top_scores[i] = -INFINITY;
        top_blocks[i] = -1;
    }

    const int64_t nsa_blocks = (seq_len + nsa_block_size - 1) / nsa_block_size;
    for (int64_t nsa_block = 0; nsa_block < nsa_blocks; ++nsa_block) {
        const int64_t tok_begin = nsa_block * nsa_block_size;
        const size_t logical_page = static_cast<size_t>(tok_begin / static_cast<int64_t>(page_block_size));
        if (logical_page >= max_num_blocks_per_seq) {
            continue;
        }
        const size_t subblock = static_cast<size_t>((tok_begin % static_cast<int64_t>(page_block_size)) / nsa_block_size);
        const Tindex physical = block_tables[seq * block_table_batch_stride + logical_page];
        const size_t cmp_block = static_cast<size_t>(physical) * subblocks_per_page + subblock;
        const size_t base_k = cmp_block * k_cmp_block_stride + kv_head * k_cmp_head_stride;
        const size_t base_v = cmp_block * v_cmp_block_stride + kv_head * v_cmp_head_stride;
        const float kd = loadFloat(k_cmp + base_k + dim);
        const float score = blockReduceSum128(qd * kd) * scale;
        if (score > top_scores[active_select_blocks - 1]) {
            int insert_pos = active_select_blocks - 1;
            while (insert_pos > 0 && score > top_scores[insert_pos - 1]) {
                top_scores[insert_pos] = top_scores[insert_pos - 1];
                top_blocks[insert_pos] = top_blocks[insert_pos - 1];
                --insert_pos;
            }
            top_scores[insert_pos] = score;
            top_blocks[insert_pos] = static_cast<int>(nsa_block);
        }
        const float vd = loadFloat(v_cmp + base_v + dim);
        const float new_m = fmaxf(comp_m, score);
        const float alpha = expf(comp_m - new_m);
        const float beta = expf(score - new_m);
        comp_acc = comp_acc * alpha + beta * vd;
        comp_l = comp_l * alpha + beta;
        comp_m = new_m;
    }
    const float comp_out = comp_l > 0.0f ? comp_acc / comp_l : 0.0f;

    float sel_acc = 0.0f;
    float sel_m = -INFINITY;
    float sel_l = 0.0f;
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API) && !defined(__MACA__) && !defined(__MACACC__)
#pragma unroll
#endif
    for (int selected = 0; selected < active_select_blocks; ++selected) {
        const int nsa_block = top_blocks[selected];
        if (nsa_block < 0) {
            continue;
        }
        const int64_t tok_begin = static_cast<int64_t>(nsa_block) * nsa_block_size;
        const int64_t tok_end = min(tok_begin + static_cast<int64_t>(nsa_block_size), seq_len);
        for (int64_t tok = tok_begin; tok < tok_end; ++tok) {
            const size_t logical_block = static_cast<size_t>(tok / static_cast<int64_t>(page_block_size));
            const size_t block_offset = static_cast<size_t>(tok % static_cast<int64_t>(page_block_size));
            if (logical_block >= max_num_blocks_per_seq) {
                continue;
            }
            const Tindex physical = block_tables[seq * block_table_batch_stride + logical_block];
            const size_t base_k = static_cast<size_t>(physical) * k_batch_stride + kv_head * k_head_stride + block_offset * k_row_stride;
            const size_t base_v = static_cast<size_t>(physical) * v_batch_stride + kv_head * v_head_stride + block_offset * v_row_stride;
            const float kd = loadFloat(k_cache + base_k + dim);
            const float score = blockReduceSum128(qd * kd) * scale;
            const float vd = loadFloat(v_cache + base_v + dim);
            const float new_m = fmaxf(sel_m, score);
            const float alpha = expf(sel_m - new_m);
            const float beta = expf(score - new_m);
            sel_acc = sel_acc * alpha + beta * vd;
            sel_l = sel_l * alpha + beta;
            sel_m = new_m;
        }
    }
    const float sel_out = sel_l > 0.0f ? sel_acc / sel_l : 0.0f;

    float win_acc = 0.0f;
    float win_m = -INFINITY;
    float win_l = 0.0f;
    const int64_t win_begin = window_size > 0 ? ((seq_len > window_size) ? (seq_len - window_size) : 0) : seq_len;
    for (int64_t tok = win_begin; tok < seq_len; ++tok) {
        const size_t logical_block = static_cast<size_t>(tok / static_cast<int64_t>(page_block_size));
        const size_t block_offset = static_cast<size_t>(tok % static_cast<int64_t>(page_block_size));
        if (logical_block >= max_num_blocks_per_seq) {
            continue;
        }
        const Tindex physical = block_tables[seq * block_table_batch_stride + logical_block];
        const size_t base_k = static_cast<size_t>(physical) * k_batch_stride + kv_head * k_head_stride + block_offset * k_row_stride;
        const size_t base_v = static_cast<size_t>(physical) * v_batch_stride + kv_head * v_head_stride + block_offset * v_row_stride;
        const float kd = loadFloat(k_cache + base_k + dim);
        const float score = blockReduceSum128(qd * kd) * scale;
        const float vd = loadFloat(v_cache + base_v + dim);
        const float new_m = fmaxf(win_m, score);
        const float alpha = expf(win_m - new_m);
        const float beta = expf(score - new_m);
        win_acc = win_acc * alpha + beta * vd;
        win_l = win_l * alpha + beta;
        win_m = new_m;
    }
    const float win_out = win_l > 0.0f ? win_acc / win_l : 0.0f;

    const float g_cmp = loadFloat(gates + seq * gates_seq_stride + 0 * gates_branch_stride + head * gates_head_stride);
    const float g_sel = loadFloat(gates + seq * gates_seq_stride + 1 * gates_branch_stride + head * gates_head_stride);
    const float g_swa = loadFloat(gates + seq * gates_seq_stride + 2 * gates_branch_stride + head * gates_head_stride);
    storeFloat(out + seq * o_stride + head * o_head_stride + dim, g_cmp * comp_out + g_sel * sel_out + g_swa * win_out);
}

} // namespace op::nsa_paged_attention::cuda

#endif // __NSA_PAGED_ATTENTION_CUDA_KERNEL_CUH__
