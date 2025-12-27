#ifndef __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
#define __PAGED_ATTENTION_PREFILL_KERNEL_CUH__

#include <cuda_fp16.h>
#include <float.h>

namespace op::paged_attention_prefill::cuda {

// =============================================================
//  Internal Helper: Block-level Reductions (Self-contained)
// =============================================================

template<typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

template<typename T, int NUM_THREADS>
struct BlockReduce {
    static __device__ __forceinline__ T max(T val) {
        static __shared__ T shared[32]; // Max 32 warps for 1024 threads
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        val = warpReduceMax(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();

        // Load back to first warp
        val = (threadIdx.x < (NUM_THREADS / 32)) ? shared[lane] : -FLT_MAX;
        
        if (wid == 0) {
            val = warpReduceMax(val);
        }
        // Broadcast result to all threads
        return __shfl_sync(0xffffffff, val, 0);
    }

    static __device__ __forceinline__ T sum(T val) {
        static __shared__ T shared[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        val = warpReduceSum(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();

        val = (threadIdx.x < (NUM_THREADS / 32)) ? shared[lane] : (T)0.0f;
        
        if (wid == 0) {
            val = warpReduceSum(val);
        }
        return __shfl_sync(0xffffffff, val, 0);
    }
};

// =============================================================
//  Main Kernel
// =============================================================

template <typename Tdata, typename Tcompute, size_t HEAD_SIZE, size_t NUM_THREADS>
__device__ void pagedAttentionPrefillKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const int32_t *block_tables_,
    const int32_t *seq_lens_,
    const int32_t *new_lens_,
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
    
    // --- 1. Coordinate Setup ---
    const int seq_idx = blockIdx.z;
    const int q_token_idx = blockIdx.y; 
    const int head_idx = blockIdx.x;
    
    const int32_t cur_new_len = new_lens_[seq_idx];
    
    if (q_token_idx >= cur_new_len) {
        return;
    }

    const int32_t total_seq_len = seq_lens_[seq_idx];
    const int32_t history_len = total_seq_len - cur_new_len;
    const int32_t global_token_idx = history_len + q_token_idx;

    const size_t num_queries_per_kv = gridDim.x / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;
    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];

    const int32_t *block_table = block_tables_ + seq_idx * max_num_blocks_per_seq;

    const Tdata *q_ptr = q_ + seq_idx * q_stride + (q_token_idx * gridDim.x + head_idx) * HEAD_SIZE;
    Tdata *out_ptr = out_ + seq_idx * o_stride + (q_token_idx * gridDim.x + head_idx) * HEAD_SIZE;

    // --- 2. Load Query to Shared Memory ---
    extern __shared__ char shared_mem_char[];
    Tcompute *shared_mem = reinterpret_cast<Tcompute *>(shared_mem_char);
    Tcompute *q_shared = shared_mem; // Size: HEAD_SIZE
    
    // [WARNING] This assumes total_seq_len fits in remaining shared memory.
    // If context length > shared memory capacity, this will crash/corrupt.
    Tcompute *logits = shared_mem + HEAD_SIZE; 
    
    for (size_t i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
        q_shared[i] = static_cast<Tcompute>(q_ptr[i]);
    }
    __syncthreads();

    // --- 3. Compute Attention Scores ---

    // --- Step 3.1: Calculate Local Max Logit (Pass 1) ---
    Tcompute local_max = -FLT_MAX;

    for (int t = threadIdx.x; t < total_seq_len; t += NUM_THREADS) {
        if (t > global_token_idx) continue;

        const int32_t b_idx = t / block_size;
        const int32_t t_off = t % block_size;
        const int32_t physical_block = block_table[b_idx];

        const Tdata *k_vec = k_cache_ + physical_block * kv_block_stride + kv_head_idx * kv_head_stride + t_off * HEAD_SIZE;

        Tcompute score = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_SIZE; ++i) {
            score += q_shared[i] * static_cast<Tcompute>(k_vec[i]);
        }
        score *= scale;
        
        if (alibi_slope != 0.0f) {
            score += alibi_slope * (t - total_seq_len + 1);
        }

        if (score > local_max) local_max = score;
    }

    // Block Reduce Max (Corrected: using internal helper)
    Tcompute global_max = BlockReduce<Tcompute, NUM_THREADS>::max(local_max);
    
    // Note: No need to store in s_global_max manually, BlockReduce broadcasts it.

    // --- Step 3.2: Calculate Exp Sum & Store Probabilities (Pass 2) ---
    // We re-compute scores to fill 'logits' (memory bounded optimization).
    // Using global_max from Step 3.1 directly.
    
    Tcompute local_sum = 0.0f;

    for (int t = threadIdx.x; t < total_seq_len; t += NUM_THREADS) {
        if (t <= global_token_idx) {
            const int32_t b_idx = t / block_size;
            const int32_t t_off = t % block_size;
            const int32_t physical_block = block_table[b_idx];
            const Tdata *k_vec = k_cache_ + physical_block * kv_block_stride + kv_head_idx * kv_head_stride + t_off * HEAD_SIZE;

            Tcompute score = 0.0f;
            #pragma unroll
            for (int i = 0; i < HEAD_SIZE; ++i) {
                score += q_shared[i] * static_cast<Tcompute>(k_vec[i]);
            }
            score *= scale;
            if (alibi_slope != 0.0f) {
                score += alibi_slope * (t - total_seq_len + 1);
            }

            // Compute Exp and Store
            Tcompute val = expf(score - global_max);
            logits[t] = val; // Storing P (unnormalized) for V pass
            local_sum += val;
        } else {
            logits[t] = 0.0f; 
        }
    }
    __syncthreads();
    
    // Block Reduce Sum (Corrected)
    Tcompute global_sum = BlockReduce<Tcompute, NUM_THREADS>::sum(local_sum);
    Tcompute inv_sum = 1.0f / (global_sum + 1e-6f);

    // --- 4. Weighted Sum V ---
    // Threads parallelize over HEAD_SIZE dimension
    for (int h = threadIdx.x; h < HEAD_SIZE; h += NUM_THREADS) {
        Tcompute acc = 0.0f;
        for (int t = 0; t <= global_token_idx; ++t) {
             const int32_t b_idx = t / block_size;
             const int32_t t_off = t % block_size;
             const int32_t physical_block = block_table[b_idx];
             
             Tcompute prob = logits[t] * inv_sum;
             
             const Tdata *v_vec = v_cache_ + physical_block * kv_block_stride + kv_head_idx * kv_head_stride + t_off * HEAD_SIZE;
             acc += prob * static_cast<Tcompute>(v_vec[h]);
        }
        out_ptr[h] = static_cast<Tdata>(acc);
    }
}

} // namespace op::paged_attention_prefill::cuda

#endif // __PAGED_ATTENTION_PREFILL_KERNEL_CUH__
