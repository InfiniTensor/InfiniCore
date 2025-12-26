#ifndef __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
#define __PAGED_ATTENTION_PREFILL_KERNEL_CUH__

namespace op::paged_attention_prefill::cuda {

template <typename Tdata, typename Tcompute,
          size_t HEAD_DIM, size_t NUM_THREADS_PER_BLOCK>
__global__ void paged_attention_prefill(
    Tdata *__restrict__ out,           // [total_seq, H, D]
    const Tdata *__restrict__ q,       // [total_seq, H, D]
    const Tdata *__restrict__ k_cache, // [num_blocks, nkvh, block_size, dh]
    const Tdata *__restrict__ v_cache,
    const int64_t *__restrict__ block_tables, // [B, max_blocks_per_seq]
    const int64_t *__restrict__ seq_lens,     // [B]
    const int64_t *__restrict__ cache_lens,   // [B]
    const int64_t *__restrict__ seq_offsets,  // [B]
    size_t num_heads,
    size_t num_kv_heads,
    size_t max_blocks_per_seq,
    size_t block_size,
    ptrdiff_t q_stride,        // = num_heads * HEAD_DIM
    ptrdiff_t kv_block_stride, // stride between physical blocks
    ptrdiff_t kv_head_stride,  // stride between kv heads
    ptrdiff_t out_stride,      // = num_heads * HEAD_DIM
    float scale) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;

    const int64_t seq_len = seq_lens[b];
    const int64_t ctx_len = cache_lens[b];
    const int64_t q_base = seq_offsets[b];

    if (h >= num_heads) {
        return;
    }

    const int kv_head = h / (num_heads / num_kv_heads);

    extern __shared__ char smem_bytes[];
    Tcompute *smem = reinterpret_cast<Tcompute *>(smem_bytes);
    Tcompute *smem_k = smem;                         // [block_size, D]
    Tcompute *smem_v = smem + block_size * HEAD_DIM; // [block_size, D]

    // Loop over prefill tokens of this request
    for (int64_t t = 0; t < seq_len; ++t) {
        const int64_t q_index = q_base + t;

        // -----------------------------
        // Load Q
        // -----------------------------
        const Tdata *q_ptr = q + q_index * q_stride + h * HEAD_DIM;

        Tcompute q_vec[HEAD_DIM];
#pragma unroll
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            q_vec[d] = static_cast<Tcompute>(q_ptr[d]);
        }

        const int64_t q_logical = ctx_len + t;
        const int64_t attn_len = q_logical + 1; // causal

        // Online softmax state
        Tcompute m = -INFINITY;
        Tcompute s = 0;
        Tcompute acc[HEAD_DIM] = {0};

        // -----------------------------
        // Iterate over KV blocks
        // -----------------------------
        for (int64_t blk = 0;
             blk * block_size < attn_len;
             ++blk) {

            const int64_t phys = block_tables[b * max_blocks_per_seq + blk];

            const int64_t base = blk * block_size;

            // -----------------------------
            // Load K/V tile into shared memory
            // -----------------------------
            for (int i = tid;
                 i < block_size * HEAD_DIM;
                 i += NUM_THREADS_PER_BLOCK) {

                const int tok = i / HEAD_DIM;
                const int d = i % HEAD_DIM;
                const int64_t logical = base + tok;

                if (logical < attn_len) {
                    const Tdata *k_ptr = k_cache
                                       + phys * kv_block_stride
                                       + kv_head * kv_head_stride
                                       + tok * HEAD_DIM;

                    const Tdata *v_ptr = v_cache
                                       + phys * kv_block_stride
                                       + kv_head * kv_head_stride
                                       + tok * HEAD_DIM;

                    smem_k[i] = static_cast<Tcompute>(k_ptr[d]);
                    smem_v[i] = static_cast<Tcompute>(v_ptr[d]);
                }
            }
            __syncthreads();

            // -----------------------------
            // Online softmax update
            // -----------------------------
            for (int tok = tid; tok < block_size; tok += NUM_THREADS_PER_BLOCK) {
                const int64_t logical = base + tok;
                if (logical < attn_len) {
                    Tcompute dot = 0;
#pragma unroll
                    for (size_t d = 0; d < HEAD_DIM; ++d) {
                        dot += q_vec[d] * smem_k[tok * HEAD_DIM + d];
                    }
                    dot *= scale;

                    const Tcompute m_new = max((float)m, (float)dot);
                    const Tcompute alpha = exp((float)(m - m_new));
                    const Tcompute beta = exp((float)(dot - m_new));

#pragma unroll
                    for (size_t d = 0; d < HEAD_DIM; ++d) {
                        acc[d] = acc[d] * alpha + beta * smem_v[tok * HEAD_DIM + d];
                    }

                    s = s * alpha + beta;
                    m = m_new;
                }
            }
            __syncthreads();
        }

        const Tcompute inv_s = Tcompute(1) / (s + Tcompute(1e-6));

        // -----------------------------
        // Write output
        // -----------------------------
        Tdata *out_ptr = out + q_index * out_stride + h * HEAD_DIM;

#pragma unroll
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            out_ptr[d] = static_cast<Tdata>(acc[d] * inv_s);
        }
    }
}
} // namespace op::paged_attention_prefill::cuda

#endif // __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
