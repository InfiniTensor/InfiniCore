#ifndef __PAGED_ATTENTION_FP8_KERNEL_CUH__
#define __PAGED_ATTENTION_FP8_KERNEL_CUH__

//================================================================================
// Paged Attention Decode Kernel for FP8(E4M3) KV Caches (clean-room)
//
// v1.6: deeper memory-level parallelism on top of the v1.5 warp-parallel scan.
// One CTA of NUM_WARPS*32 threads (default 8 warps = 256 threads) per
// (sequence, query head); warp `w` visits tokens w, w+NUM_WARPS, ... of every
// referenced page (per-page striding; the global token index t = t_base + tb
// keeps ALiBi correct for any page size). Each lane owns DL = HEAD_SIZE/32
// consecutive head dims (4 for hd128, 2 for hd64), so a single uint32/uint16
// load fetches all of the lane's E4M3 codes for one token and the warp covers
// HEAD_SIZE contiguous bytes per K/V row.
//
// The token stream is walked with a (logical block, token-in-block) cursor and
// register double buffering: while token i is being processed, the packed K/V
// words and both per-token scales of token i+1 are already in flight (loads
// issued one iteration ahead, across page boundaries). The qk dot product is
// a shuffle-only warp reduction; every warp keeps its own online-softmax state
// (m, l, acc[DL]) in registers; the partial states are merged once at the end
// through shared memory (the only __syncthreads in the kernel), rescaling by
// exp2(m_w - m_total) in the standard online-softmax fashion.
//
// Semantics are unchanged from v1: dequant-on-load
//   x = e4m3_decode(code) * scale[physical_block, kv_head, slot]
// with per-token-per-kv-head F32 scales written by paged_caching, online
// softmax in the log2 domain (scale * log2e, optional ALiBi, final division
// by l + 1e-6), F16/BF16 output, HEAD_SIZE in {64, 128}.
//
// v2: optional cross-CTA split-kv (flash-decoding). When num_splits > 1 the
// grid gains a z dimension (split_idx = blockIdx.z) and each CTA scans only
// the contiguous token shard [split_idx*shard, min(seq_len, +shard)) of its
// (sequence, query head); instead of the final output it writes the merged
// per-CTA state (m, l, unnormalized acc) to workspace partials, laid out as
//   partial_acc [num_splits, num_seqs, num_heads, HEAD_SIZE]
//   partial_m/l [num_splits, num_seqs, num_heads]
// (same convention as the F16/BF16 family in kernel_v2.cuh). A second
// combine kernel (one CTA of HEAD_SIZE threads per (seq, head)) then merges
// the shards in the log2 domain and divides by l + 1e-6. Shards with no
// token produce the neutral element (m = -inf, l = 0, acc = 0).
//
// Alignment note: the packed code loads assume each K/V row is DL-byte
// aligned, i.e. the cache base pointer and the batch/head/row strides are
// multiples of DL bytes. This holds for any contiguous cache pool (row stride
// == HEAD_SIZE) and for block/head slices of one.
//================================================================================

#include <cstdint>
#include <type_traits>

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

namespace op::paged_attention::cuda {

// Number of warps per CTA in the FP8 decode kernel. Tunable (4/8/16); the
// launcher sizes the block from this constant, so the two stay in sync.
constexpr int kFp8DecodeNumWarps = 32;

// Upper bound on cross-CTA split-kv shards; the launcher clamps num_splits to
// this and sizes the workspace for it.
constexpr int kFp8DecodeMaxSplits = 8;

template <typename Tindex, typename Tdata, int HEAD_SIZE>
__device__ void flashAttentionDecodeFp8Kernel(
    Tdata *out_,
    float *partial_acc_, // [num_splits, num_seqs, num_heads, HEAD_SIZE]; nullptr => no split-kv
    float *partial_m_,   // [num_splits, num_seqs, num_heads]
    float *partial_l_,   // [num_splits, num_seqs, num_heads]
    const Tdata *q_,
    const uint8_t *k_cache_,
    const uint8_t *v_cache_,
    const float *k_scale_,
    const float *v_scale_,
    const Tindex *block_tables_,
    const Tindex *cache_lens_,
    const float *alibi_slopes_,
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

    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128,
                  "FP8 decode kernel supports head_size 64/128 only.");

    constexpr int DL = HEAD_SIZE / 32; // head dims per lane: 4 (hd128) or 2 (hd64)
    using PackT = std::conditional_t<DL == 4, uint32_t, uint16_t>;
    constexpr int NUM_WARPS = kFp8DecodeNumWarps;

    const size_t seq_idx = blockIdx.y;
    const size_t head_idx = blockIdx.x;
    const int split_idx = static_cast<int>(blockIdx.z);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int seq_len = static_cast<int>(cache_lens_[seq_idx]);
    if (seq_len <= 0) {
        return;
    }

    // This CTA's contiguous token shard [tok_lo, tok_hi). Without split-kv
    // (num_splits == 1) the shard is the whole sequence.
    const int shard = (seq_len + num_splits - 1) / num_splits;
    const int tok_lo = min(split_idx * shard, seq_len);
    const int tok_hi = min(seq_len, tok_lo + shard);

    const size_t num_heads = gridDim.x;
    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    const Tindex *block_table = block_tables_ + seq_idx * max_num_blocks_per_seq;

    // This lane's head dims are d0 .. d0+DL-1 (contiguous => packed byte loads).
    const int d0 = lane * DL;
    const Tdata *q_ptr = q_ + seq_idx * q_stride + head_idx * q_head_stride + d0;
    float q_reg[DL];
#pragma unroll
    for (int j = 0; j < DL; ++j) {
        q_reg[j] = static_cast<float>(q_ptr[j]);
    }

    // Per-warp online softmax state; acc holds this lane's DL dims.
    float acc[DL];
#pragma unroll
    for (int j = 0; j < DL; ++j) {
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    const int pbs = static_cast<int>(page_block_size);

    // This CTA's token window within logical block `lb`, intersected with the
    // split shard: [tokenBegin, tokenEnd). tokenBegin is nonzero only on the
    // shard's first page; the last page of the shard may be partial.
    auto tokenBegin = [&](int lb) -> int {
        return max(tok_lo - lb * pbs, 0);
    };
    auto tokenEnd = [&](int lb) -> int {
        return min(pbs, tok_hi - lb * pbs);
    };

    // One token's prefetched payload: the lane's packed K/V code words plus
    // the (warp-uniform) per-token dequant scales.
    struct KvPack {
        PackT k, v;
        float ks, vs;
    };

    // Load token `tb` of logical block `lb` into `pack`. All four loads are
    // independent; the block_table read is L1-cached and shared by all warps.
    auto loadToken = [&](int lb, int tb, KvPack &pack) {
        const ptrdiff_t physical_block = static_cast<ptrdiff_t>(block_table[lb]);
        const uint8_t *k_base = k_cache_ + physical_block * k_batch_stride + static_cast<ptrdiff_t>(kv_head_idx) * k_head_stride;
        const uint8_t *v_base = v_cache_ + physical_block * v_batch_stride + static_cast<ptrdiff_t>(kv_head_idx) * v_head_stride;
        const float *k_scale_base = k_scale_ + physical_block * k_scale_block_stride + static_cast<ptrdiff_t>(kv_head_idx) * k_scale_head_stride;
        const float *v_scale_base = v_scale_ + physical_block * v_scale_block_stride + static_cast<ptrdiff_t>(kv_head_idx) * v_scale_head_stride;
        pack.k = *reinterpret_cast<const PackT *>(k_base + tb * k_row_stride + d0);
        pack.v = *reinterpret_cast<const PackT *>(v_base + tb * v_row_stride + d0);
        pack.ks = k_scale_base[tb * k_scale_slot_stride];
        pack.vs = v_scale_base[tb * v_scale_slot_stride];
    };

    // Per-warp token cursor: (logical block, token-in-block). Within each
    // page's window warp `warp` owns tokens tokenBegin+warp,
    // tokenBegin+warp+NUM_WARPS, ...; pages with no token for this warp are
    // skipped. The shard bounds make this equivalent to the v1 full-sequence
    // walk when num_splits == 1.
    int cur_lb = tok_lo / pbs;
    int cur_tb = tokenBegin(cur_lb) + warp;
    while (cur_lb * pbs < tok_hi && cur_tb >= tokenEnd(cur_lb)) {
        ++cur_lb;
        cur_tb = tokenBegin(cur_lb) + warp;
    }
    bool has_cur = cur_lb * pbs < tok_hi;

    KvPack cur{}, nxt{};
    if (has_cur) {
        loadToken(cur_lb, cur_tb, cur);
    }

    // Software-pipelined scan: process `cur` while `nxt` is being fetched.
    while (has_cur) {
        // Advance the cursor (possibly across page boundaries) and issue the
        // next token's loads before touching the current payload.
        int nxt_lb = cur_lb;
        int nxt_tb = cur_tb + NUM_WARPS;
        if (nxt_tb >= tokenEnd(nxt_lb)) {
            ++nxt_lb;
            nxt_tb = tokenBegin(nxt_lb) + warp;
            while (nxt_lb * pbs < tok_hi && nxt_tb >= tokenEnd(nxt_lb)) {
                ++nxt_lb;
                nxt_tb = tokenBegin(nxt_lb) + warp;
            }
        }
        const bool has_nxt = nxt_lb * pbs < tok_hi;
        if (has_nxt) {
            loadToken(nxt_lb, nxt_tb, nxt);
        }

        const int t = cur_lb * pbs + cur_tb;

        float partial = 0.0f;
#pragma unroll
        for (int j = 0; j < DL; ++j) {
            const uint8_t code = static_cast<uint8_t>((cur.k >> (8 * j)) & 0xFF);
            partial += q_reg[j] * (infiniopFp8E4m3Decode(code) * cur.ks);
        }
        // Shuffle-only warp reduction (no smem, no __syncthreads).
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset);
        }
        const float qk = partial;

        float score = qk * scale_log2;
        if (alibi_slope != 0.0f) {
            score += (alibi_slope * static_cast<float>(t - (seq_len - 1))) * kLog2e;
        }
        const float m_new = fmaxf(m, score);
        const float alpha = exp2f(m - m_new);
        const float beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;

#pragma unroll
        for (int j = 0; j < DL; ++j) {
            const uint8_t code = static_cast<uint8_t>((cur.v >> (8 * j)) & 0xFF);
            acc[j] = acc[j] * alpha + beta * (infiniopFp8E4m3Decode(code) * cur.vs);
        }

        cur = nxt;
        cur_lb = nxt_lb;
        cur_tb = nxt_tb;
        has_cur = has_nxt;
    }

    // ---- Cross-warp merge (the only block-wide synchronization) ----
    __shared__ float m_part[NUM_WARPS];
    __shared__ float l_part[NUM_WARPS];
    __shared__ float acc_part[NUM_WARPS][HEAD_SIZE];

    if (lane == 0) {
        m_part[warp] = m;
        l_part[warp] = l;
    }
#pragma unroll
    for (int j = 0; j < DL; ++j) {
        acc_part[warp][d0 + j] = acc[j];
    }
    __syncthreads();

    // Scalar (m, l) merge, computed redundantly by all threads. A warp that
    // saw no token has m = -inf / l = 0 and contributes weight 0. The explicit
    // -inf guard (instead of relying on exp2f(-inf - m_total) == 0) also keeps
    // an all-empty split shard — where m_total itself is -inf and
    // exp2f(-inf - -inf) would be NaN — at the neutral element (0, 0, 0).
    float m_total = m_part[0];
#pragma unroll
    for (int w = 1; w < NUM_WARPS; ++w) {
        m_total = fmaxf(m_total, m_part[w]);
    }
    float wgt[NUM_WARPS];
    float l_total = 0.0f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
        wgt[w] = (m_part[w] == -INFINITY) ? 0.0f : exp2f(m_part[w] - m_total);
        l_total += l_part[w] * wgt[w];
    }

    // HEAD_SIZE output dims over the CTA: thread `tid` writes dim `tid`
    // (threads with tid >= HEAD_SIZE idle out).
    const int tid = threadIdx.x;

    // Split-kv: publish this shard's merged (m, l, unnormalized acc) and let
    // the combine kernel produce the final output.
    if (partial_m_ != nullptr) {
        const size_t n = gridDim.y * gridDim.x; // num_seqs * num_heads
        const size_t idx = static_cast<size_t>(split_idx) * n + seq_idx * gridDim.x + head_idx;
        if (tid == 0) {
            partial_m_[idx] = (l_total > 0.0f) ? m_total : -INFINITY;
            partial_l_[idx] = l_total;
        }
        if (tid < HEAD_SIZE) {
            float o = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                o += acc_part[w][tid] * wgt[w];
            }
            partial_acc_[idx * HEAD_SIZE + tid] = o;
        }
        return;
    }

    const float inv_l = 1.0f / (l_total + 1e-6f);
    if (tid < HEAD_SIZE) {
        float o = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            o += acc_part[w][tid] * wgt[w];
        }
        o *= inv_l;
        Tdata *out_ptr = out_ + seq_idx * o_stride + head_idx * o_head_stride + tid;
        if constexpr (std::is_same_v<Tdata, half>) {
            *out_ptr = __float2half_rn(o);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            *out_ptr = __float2bfloat16_rn(o);
        } else {
            *out_ptr = static_cast<Tdata>(o);
        }
    }
}

// Cross-CTA split-kv combine: one CTA of HEAD_SIZE threads per (sequence,
// query head) merges the per-shard partials in the log2 domain and writes the
// final output. Mirrors the FP8 kernel's own cross-warp merge; empty shards
// carry m = -inf / l = 0 and contribute weight 0 (split 0 is never empty for
// seq_len > 0, so m_total stays finite).
template <typename Tdata, int HEAD_SIZE>
__device__ void flashAttentionDecodeFp8SplitKvCombineKernel(
    Tdata *out_,
    const float *partial_acc_, // [num_splits, num_seqs, num_heads, HEAD_SIZE]
    const float *partial_m_,   // [num_splits, num_seqs, num_heads]
    const float *partial_l_,   // [num_splits, num_seqs, num_heads]
    int num_splits,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {

    const size_t seq_idx = blockIdx.y;
    const size_t head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const size_t n = gridDim.y * gridDim.x; // num_seqs * num_heads
    const size_t base = seq_idx * gridDim.x + head_idx;

    // Scalar (m, l) merge, computed redundantly by all threads.
    float m_total = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        m_total = fmaxf(m_total, partial_m_[s * n + base]);
    }
    float wgt[kFp8DecodeMaxSplits];
    float l_total = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        const float ms = partial_m_[s * n + base];
        wgt[s] = (ms == -INFINITY) ? 0.0f : exp2f(ms - m_total);
        l_total += partial_l_[s * n + base] * wgt[s];
    }
    const float inv_l = 1.0f / (l_total + 1e-6f);

    if (tid < HEAD_SIZE) {
        float o = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            o += partial_acc_[(s * n + base) * HEAD_SIZE + tid] * wgt[s];
        }
        o *= inv_l;
        Tdata *out_ptr = out_ + seq_idx * o_stride + head_idx * o_head_stride + tid;
        if constexpr (std::is_same_v<Tdata, half>) {
            *out_ptr = __float2half_rn(o);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            *out_ptr = __float2bfloat16_rn(o);
        } else {
            *out_ptr = static_cast<Tdata>(o);
        }
    }
}

} // namespace op::paged_attention::cuda

#endif // __PAGED_ATTENTION_FP8_KERNEL_CUH__
