#ifndef __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
#define __PAGED_ATTENTION_PREFILL_KERNEL_CUH__

#include "../../../reduce/cuda/reduce.cuh"

namespace op::paged_attention_prefill::cuda {

template <typename Tdata, typename Tcompute, size_t HEAD_SIZE, size_t NUM_THREADS>
__device__ void pagedAttentionPrefillKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const int32_t *block_tables_,
    const int32_t *seq_lens_,
    const float *alibi_slopes_,
    const size_t num_kv_heads,
    const float scale,
    const size_t max_num_blocks_per_seq,
    const size_t block_size,
    const size_t max_new_len,
    const ptrdiff_t q_stride,
    const ptrdiff_t kv_block_stride,
    const ptrdiff_t kv_head_stride,
    const ptrdiff_t o_stride) {

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int num_heads = gridDim.x;
    const int32_t total_seq_len = seq_lens_[seq_idx];
    
    if (total_seq_len == 0) return;

    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;
    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];

    const int32_t *block_table = block_tables_ + seq_idx * max_num_blocks_per_seq;

    // 共享内存布局: [Q_shared(HEAD_SIZE) | Logits(total_seq_len)]
    extern __shared__ char shared_mem_char[];
    Tcompute *shared_mem = reinterpret_cast<Tcompute *>(shared_mem_char);
    Tcompute *q_shared = shared_mem;
    Tcompute *logits = shared_mem + HEAD_SIZE;

    // Prefill 阶段：循环处理当前 Sequence 中的每一个新 Query Token
    for (size_t q_token_idx = 0; q_token_idx < max_new_len; ++q_token_idx) {
        // 当前 Query Token 在全局 Sequence 中的逻辑位置
        const int32_t query_pos = total_seq_len - max_new_len + q_token_idx;
        if (query_pos < 0) continue;

        // 1. Load Q for current token
        const Tdata *q_ptr = q_ + seq_idx * q_stride + q_token_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE;
        for (size_t i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
            q_shared[i] = static_cast<Tcompute>(q_ptr[i]);
        }
        __syncthreads();

        // 2. Compute QK Dot Product with Causal Mask
        // 只能看到位置 <= query_pos 的 Key
        for (size_t token_idx = threadIdx.x; token_idx <= query_pos; token_idx += NUM_THREADS) {
            const int32_t b_idx = token_idx / block_size;
            const int32_t t_in_b_idx = token_idx % block_size;
            const int32_t p_block_num = block_table[b_idx];

            const Tdata *k_vec_ptr = k_cache_ + p_block_num * kv_block_stride + 
                                     kv_head_idx * kv_head_stride + t_in_b_idx * HEAD_SIZE;

            Tcompute qk = 0.0f;
            #pragma unroll
            for (size_t i = 0; i < HEAD_SIZE; ++i) {
                qk += q_shared[i] * static_cast<Tcompute>(k_vec_ptr[i]);
            }
            qk *= scale;
            if (alibi_slope != 0.0f) {
                qk += alibi_slope * (static_cast<int>(token_idx) - query_pos);
            }
            logits[token_idx] = qk;
        }
        __syncthreads();

        // 3. Softmax (Max -> Exp -> Sum -> Inv)
        const int32_t current_attn_len = query_pos + 1;
        Tcompute qk_max = op::common_cuda::reduce_op::max<NUM_THREADS, Tcompute>(logits, current_attn_len);
        
        for (size_t i = threadIdx.x; i < current_attn_len; i += NUM_THREADS) {
            logits[i] = expf(logits[i] - qk_max);
        }
        __syncthreads();

        Tcompute exp_sum = op::common_cuda::reduce_op::sum<NUM_THREADS, Tcompute, Tcompute>(logits, current_attn_len);
        Tcompute inv_sum = 1.0f / (exp_sum + 1e-6f);

        // 4. Aggregate Values (V)
        Tdata *out_ptr = out_ + seq_idx * o_stride + q_token_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE;
        
        for (size_t h_dim = threadIdx.x; h_dim < HEAD_SIZE; h_dim += NUM_THREADS) {
            Tcompute acc = 0.0f;
            for (size_t token_idx = 0; token_idx < current_attn_len; ++token_idx) {
                const int32_t b_idx = token_idx / block_size;
                const int32_t t_in_b_idx = token_idx % block_size;
                const int32_t p_block_num = block_table[b_idx];
                
                const Tdata *v_vec_ptr = v_cache_ + p_block_num * kv_block_stride + 
                                         kv_head_idx * kv_head_stride + t_in_b_idx * HEAD_SIZE;
                
                acc += logits[token_idx] * inv_sum * static_cast<Tcompute>(v_vec_ptr[h_dim]);
            }
            out_ptr[h_dim] = static_cast<Tdata>(acc);
        }
        __syncthreads(); // 准备下一个 Q token
    }
}

} // namespace op::paged_attention_prefill::cuda

#endif
