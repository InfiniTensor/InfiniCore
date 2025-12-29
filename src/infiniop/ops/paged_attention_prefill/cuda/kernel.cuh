#ifndef __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
#define __PAGED_ATTENTION_PREFILL_KERNEL_CUH__

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

namespace op::paged_attention_prefill::cuda {

// =============================================================
//  Naive Kernel: No Shared Memory, No Warp Shuffle
//  Correctness prioritized.
//  Changed to __global__ so it can be launched directly.
// =============================================================

template <typename Tdata, typename Tcompute>
__global__ void pagedAttentionPrefillKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const int32_t *block_tables_,
    const int32_t *seq_lens_,
    const int32_t *new_lens_,
    const float *alibi_slopes_,
    const size_t num_heads,
    const size_t num_kv_heads,
    const float scale,
    const size_t max_num_blocks_per_seq,
    const size_t block_size,
    const size_t max_new_len,
    const ptrdiff_t q_stride,
    const ptrdiff_t kv_block_stride,
    const ptrdiff_t kv_head_stride,
    const ptrdiff_t o_stride,
    const size_t head_size_const) {

    // --- 1. Coordinate Setup ---
    // Grid: (num_heads, max_new_len, num_seqs)
    // Block: (HEAD_SIZE, 1, 1)
    const int seq_idx = blockIdx.z;
    const int q_token_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;

    // Check boundary
    const int32_t cur_new_len = new_lens_[seq_idx];
    if (q_token_idx >= cur_new_len) {
        return;
    }

    // Safety check for head size
    if (dim_idx >= head_size_const) {
        return;
    }

    // Dimensions
    const int32_t total_seq_len = seq_lens_[seq_idx];
    const int32_t history_len = total_seq_len - cur_new_len;
    const int32_t global_token_idx = history_len + q_token_idx;

    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;

    const int32_t *block_table = block_tables_ + seq_idx * max_num_blocks_per_seq;

    // 假设 q_stride 传入的是单个 Sequence 在内存中占据的 Tdata 数量 (即 max_new_len * num_heads * head_size)
    const Tdata *q_ptr_base = q_ + seq_idx * q_stride + q_token_idx * (num_heads * head_size_const) + head_idx * head_size_const;

    // --- 2. 修改 Out 的基地址计算 ---
    Tdata *out_ptr = out_ + seq_idx * o_stride + q_token_idx * (num_heads * head_size_const) + head_idx * head_size_const;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];

    // // 只让第一个 Sequence, 第一个 Token, 第一个 Head 的第一个线程执行打印
    // if (seq_idx == 0 && q_token_idx == 0 && head_idx == 0 && dim_idx == 0) {
    //     printf("DEBUG: Scale=%f, HeadSize=%zu, BlockSize=%zu\n", scale, head_size_const, block_size);

    //     // 检查 Q 的前 5 个元素
    //     for(int i=0; i<5; ++i) printf("Q[%d]=%f ", i, (float)q_ptr_base[i]);
    //     printf("\n");

    //     // 检查第一个 KV Block 的前 5 个元素
    //     const int32_t first_physical_block = block_table[0];
    //     const Tdata *first_k = k_cache_ + first_physical_block * kv_block_stride;
    //     for(int i=0; i<5; ++i) printf("K_cache[0][%d]=%f ", i, (float)first_k[i]);
    //     printf("\n");
    // }

    // --- Pass 1: Find Global Max ---
    Tcompute max_score = -FLT_MAX;

    for (int t = 0; t < total_seq_len; ++t) {
        if (t > global_token_idx) {
            break;
        }

        const int32_t b_idx = t / block_size;
        const int32_t t_off = t % block_size;
        const int32_t physical_block = block_table[b_idx];

        const Tdata *k_vec = k_cache_ + physical_block * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size_const;

        Tcompute score = 0.0f;
        for (int d = 0; d < head_size_const; ++d) {
            score += static_cast<Tcompute>(q_ptr_base[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= scale;

        if (alibi_slope != 0.0f) {
            score += alibi_slope * (t - total_seq_len + 1);
        }

        if (score > max_score) {
            max_score = score;
        }
    }

    // --- Pass 2: Calculate Denominator (Sum Exp) ---
    Tcompute sum_exp = 0.0f;

    for (int t = 0; t < total_seq_len; ++t) {
        if (t > global_token_idx) {
            break;
        }

        const int32_t b_idx = t / block_size;
        const int32_t t_off = t % block_size;
        const int32_t physical_block = block_table[b_idx];
        const Tdata *k_vec = k_cache_ + physical_block * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size_const;

        Tcompute score = 0.0f;
        for (int d = 0; d < head_size_const; ++d) {
            score += static_cast<Tcompute>(q_ptr_base[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= scale;
        if (alibi_slope != 0.0f) {
            score += alibi_slope * (t - total_seq_len + 1);
        }

        sum_exp += expf(score - max_score);
    }

    // --- Pass 3: Calculate Weighted Sum (V) ---
    Tcompute acc = 0.0f;
    Tcompute inv_sum = 1.0f / (sum_exp + 1e-6f);

    for (int t = 0; t < total_seq_len; ++t) {
        if (t > global_token_idx) {
            break;
        }

        const int32_t b_idx = t / block_size;
        const int32_t t_off = t % block_size;
        const int32_t physical_block = block_table[b_idx];

        // Re-compute Score
        const Tdata *k_vec = k_cache_ + physical_block * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size_const;
        Tcompute score = 0.0f;
        for (int d = 0; d < head_size_const; ++d) {
            score += static_cast<Tcompute>(q_ptr_base[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= scale;
        if (alibi_slope != 0.0f) {
            score += alibi_slope * (t - total_seq_len + 1);
        }

        Tcompute prob = expf(score - max_score) * inv_sum;

        const Tdata *v_vec = v_cache_ + physical_block * kv_block_stride + kv_head_idx * kv_head_stride + t_off * head_size_const;

        acc += prob * static_cast<Tcompute>(v_vec[dim_idx]);
    }

    out_ptr[dim_idx] = static_cast<Tdata>(acc);
}

} // namespace op::paged_attention_prefill::cuda

#endif // __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
