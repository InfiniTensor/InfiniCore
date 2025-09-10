// kernel.cuh (in op/chunk_gated_delta_rule/cuda/)

#ifndef __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__
#define __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__

#include <cuda_fp16.h>
#include <cmath>
// Tdata:  (e.g., half)
// Tcompute:  (e.g., float)
template <typename Tdata, typename Tcompute, size_t Dk, size_t Dv, size_t NUM_THREADS>
__device__ void chunkGatedDeltaRuleKernel(
    Tdata* out,
    Tdata* final_state,
    const Tdata* q,
    const Tdata* k,
    const Tdata* v,
    const Tdata* g,
    const Tdata* beta,
    const Tdata* initial_state,
    bool use_qk_l2norm
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;
    
    // T=1 for decode stage, so seq_idx is always 0
    const int seq_idx = 0;

    const size_t H = gridDim.y;
    const size_t base_offset_qkv = (batch_idx * H + head_idx) * Dk; // T=1, Dk=Dv for simplicity now
    const size_t base_offset_gb = (batch_idx * H + head_idx); // T=1
    const size_t state_offset = (batch_idx * H + head_idx) * Dk * Dv;
    
    const Tdata* q_ptr = q + base_offset_qkv;
    const Tdata* k_ptr = k + base_offset_qkv;
    const Tdata* v_ptr = v + base_offset_qkv; // Assuming Dv = Dk
    const Tdata* g_ptr = g + base_offset_gb;
    const Tdata* beta_ptr = beta + base_offset_gb;
    const Tdata* initial_state_ptr = initial_state + state_offset;

    Tdata* out_ptr = out + base_offset_qkv;
    Tdata* final_state_ptr = final_state + state_offset;

    extern __shared__ char shared_mem_char[];
    Tcompute* shared_mem = reinterpret_cast<Tcompute*>(shared_mem_char);
    
    // shared memory layout: q_local[Dk], k_local[Dk], norm_val[1]
    Tcompute* q_local = shared_mem;
    Tcompute* k_local = q_local + Dk;
    Tcompute* norm_val = k_local + Dk; // for reduction

    // 1. Load Q and K into shared memory and optionally normalize
    // Load
    for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
        q_local[i] = static_cast<Tcompute>(q_ptr[i]);
        k_local[i] = static_cast<Tcompute>(k_ptr[i]);
    }

    if (use_qk_l2norm) {
        __syncthreads();
        // Parallel reduction to compute L2 norm for Q
        Tcompute sum_sq = 0.0f;
        for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
            sum_sq += q_local[i] * q_local[i];
        }
        // Simplified reduction, for real use CUB will be better
        // This part needs a proper block-wide reduction implementation
        norm_val[thread_idx] = sum_sq;
        __syncthreads();
        if (thread_idx == 0) {
            Tcompute total_sum_sq = 0.0f;
            for(int i=0; i<NUM_THREADS; ++i) total_sum_sq += norm_val[i];
            norm_val[0] = rsqrtf(total_sum_sq + 1e-6f);
        }
        __syncthreads();
        Tcompute r_norm_q = norm_val[0];
        
        // Normalize Q
        for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
            q_local[i] *= r_norm_q;
        }

        // Repeat for K
        sum_sq = 0.0f;
        for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
            sum_sq += k_local[i] * k_local[i];
        }
        norm_val[thread_idx] = sum_sq;
        __syncthreads();
        if (thread_idx == 0) {
            Tcompute total_sum_sq = 0.0f;
            for(int i=0; i<NUM_THREADS; ++i) total_sum_sq += norm_val[i];
            norm_val[0] = rsqrtf(total_sum_sq + 1e-6f);
        }
        __syncthreads();
        Tcompute r_norm_k = norm_val[0];

        // Normalize K
        for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
            k_local[i] *= r_norm_k;
        }
        __syncthreads();
    }
    
    // 2. Perform the recurrent update logic
    Tcompute g_t = expf(static_cast<Tcompute>(*g_ptr));
    Tcompute beta_t = static_cast<Tcompute>(*beta_ptr);
    Tcompute scale = rsqrtf(static_cast<Tcompute>(Dk));

    for (int i = thread_idx; i < Dk; i+= NUM_THREADS) {
        q_local[i] *= scale;
    }
    __syncthreads();

    // Loop over Dv, each thread computes an element of the delta and output vector
    for (int dv_idx = thread_idx; dv_idx < Dv; dv_idx += NUM_THREADS) {
        Tcompute kv_mem = 0.0f;
        // Calculate kv_mem = sum(h_{t-1} * k_t)
        for (int dk_idx = 0; dk_idx < Dk; ++dk_idx) {
            Tcompute h_prev = static_cast<Tcompute>(initial_state_ptr[dk_idx * Dv + dv_idx]);
            kv_mem += (h_prev * g_t) * k_local[dk_idx];
        }

        Tcompute v_t = static_cast<Tcompute>(v_ptr[dv_idx]);
        Tcompute delta = (v_t - kv_mem) * beta_t;

        // Calculate final state h_t = h_{t-1} * g + k_t * delta
        // And write it back
        for (int dk_idx = 0; dk_idx < Dk; ++dk_idx) {
             Tcompute h_prev = static_cast<Tcompute>(initial_state_ptr[dk_idx * Dv + dv_idx]);
             Tcompute h_final = (h_prev * g_t) + (k_local[dk_idx] * delta);
             final_state_ptr[dk_idx * Dv + dv_idx] = static_cast<Tdata>(h_final);
        }
        
        // Calculate output o_t = sum(h_t * q_t)
        // This requires another reduction. For simplicity, we assume one thread calculates one output element.
        // A more optimized version would have all threads collaborating.
    }
    __syncthreads(); // Ensure final_state is fully written

    // All threads collaborate to compute the final output vector
    for (int dv_idx = thread_idx; dv_idx < Dv; dv_idx += NUM_THREADS) {
        Tcompute out_val = 0.0f;
        for (int dk_idx = 0; dk_idx < Dk; ++dk_idx) {
            out_val += static_cast<Tcompute>(final_state_ptr[dk_idx * Dv + dv_idx]) * q_local[dk_idx];
        }
        out_ptr[dv_idx] = static_cast<Tdata>(out_val);
    }
}

#endif // __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__