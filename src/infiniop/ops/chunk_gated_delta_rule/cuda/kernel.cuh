// op/chunk_gated_delta_rule/cuda/kernel.cuh

#ifndef __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__
#define __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__

#include <cuda_fp16.h>
#include <cub/cub.cuh>

// Define a vectorized type for loading/storing multiple half-precision floats.
// This assumes Dk and Dv are multiples of 8.
// using half8 = __align__(16) struct { half v[8]; };

template <typename Tdata, typename Tcompute,
          size_t Dk, size_t Dv, size_t BLOCK_THREADS>
__device__ void chunkGatedDeltaRuleKernel(
    Tdata* out,
    Tdata* final_state,
    const Tdata* q,
    const Tdata* k,
    const Tdata* v,
    const Tdata* g,
    const Tdata* beta,
    const Tdata* initial_state,
    bool use_qk_l2norm,
    const size_t chunk_size,
    const size_t T // Original sequence length, must be passed from host
) {
    // Grid Strategy: Each block handles one sequence for one head.
    // gridDim.x = B, gridDim.y = H
    const size_t batch_idx = blockIdx.x;
    const size_t head_idx = blockIdx.y;
    const size_t thread_idx = threadIdx.x;

    const size_t B = gridDim.x;
    const size_t H = gridDim.y;

    const size_t T_padded = (T + chunk_size - 1) / chunk_size * chunk_size;
    const size_t num_chunks = T_padded / chunk_size;
    const float scale = rsqrtf(static_cast<Tcompute>(Dk));

    using BlockScan = cub::BlockScan<Tcompute, BLOCK_THREADS>;
    
    // --- Shared Memory Layout ---
    extern __shared__ char shared_mem_char[];
    Tcompute* shared_mem = reinterpret_cast<Tcompute*>(shared_mem_char);
    
    Tcompute* q_s = shared_mem;
    Tcompute* k_s = q_s + chunk_size * Dk;
    Tcompute* v_s = k_s + chunk_size * Dk;
    Tcompute* g_s = v_s + chunk_size * Dv;
    Tcompute* beta_s = g_s + chunk_size;
    Tcompute* g_cumsum_s = beta_s + chunk_size;
    Tcompute* decay_mask_s = g_cumsum_s + chunk_size;
    Tcompute* attn_s = decay_mask_s + chunk_size * chunk_size;
    Tcompute* k_cumdecay_s = attn_s + chunk_size * chunk_size;
    Tcompute* value_prime_s = k_cumdecay_s + chunk_size * Dk;
    Tcompute* v_prime_s = value_prime_s + chunk_size * Dv;
    Tcompute* attn_inter_s = v_prime_s + chunk_size * Dv;
    Tcompute* v_new_s = attn_inter_s + chunk_size * Dv;
    
    typename BlockScan::TempStorage* scan_storage = (typename BlockScan::TempStorage*)(v_new_s + chunk_size * Dv);
    Tcompute* inter_chunk_state_s = (Tcompute*)(scan_storage + 1);

    // === Phase 1: Initialize inter-chunk state from initial_state ===
    const Tdata* initial_state_ptr = initial_state + (batch_idx * H + head_idx) * (Dk * Dv);
    for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
        inter_chunk_state_s[i] = static_cast<Tcompute>(initial_state_ptr[i]);
    }

    // --- Main loop over chunks of the sequence ---
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        __syncthreads(); // Ensure state is ready and previous chunk's writes are visible
        size_t chunk_offset = chunk_idx * chunk_size;

        // --- 2.1: Collaborative Loading of chunk data ---
        for (size_t i = thread_idx; i < chunk_size; i += BLOCK_THREADS) {
            size_t t_idx = chunk_offset + i;
            ptrdiff_t gb_offset = (batch_idx * T * H) + (t_idx * H) + head_idx;
            g_s[i] = (t_idx < T) ? static_cast<Tcompute>(g[gb_offset]) : 0.0f;
            beta_s[i] = (t_idx < T) ? static_cast<Tcompute>(beta[gb_offset]) : 1.0f;
        }

        for (size_t i = thread_idx; i < chunk_size * Dk; i += BLOCK_THREADS) {
            size_t t_local = i / Dk;
            size_t d = i % Dk;
            size_t t_global = chunk_offset + t_local;
            ptrdiff_t qkv_offset = (batch_idx * T * H * Dk) + (t_global * H * Dk) + (head_idx * Dk) + d;

            q_s[i] = (t_global < T) ? static_cast<Tcompute>(q[qkv_offset]) : 0.0f;
            k_s[i] = (t_global < T) ? static_cast<Tcompute>(k[qkv_offset]) : 0.0f;
        }
        for (size_t i = thread_idx; i < chunk_size * Dv; i += BLOCK_THREADS) {
             size_t t_local = i / Dv;
             size_t d = i % Dv;
             size_t t_global = chunk_offset + t_local;
             ptrdiff_t v_offset = (batch_idx * T * H * Dv) + (t_global * H * Dv) + (head_idx * Dv) + d;
             v_s[i] = (t_global < T) ? static_cast<Tcompute>(v[v_offset]) : 0.0f;
        }
        __syncthreads();
        
        // --- 2.2: Optional L2 Normalization ---
        if (use_qk_l2norm) {
            // Each thread t normalizes its own q_t and k_t vectors.
                for (size_t t = thread_idx; t < chunk_size; t += BLOCK_THREADS) {
                size_t t_global = chunk_offset + t;
                if (t_global < T) {
                    Tcompute q_norm_sq = 0.0f;
                    Tcompute k_norm_sq = 0.0f;
                    
                    // Calculate sum of squares for q_t and k_t
                    for (size_t d = 0; d < Dk; ++d) {
                        Tcompute q_val = q_s[t * Dk + d];
                        Tcompute k_val = k_s[t * Dk + d];
                        q_norm_sq += q_val * q_val;
                        k_norm_sq += k_val * k_val;
                    }

                    // Calculate inverse sqrt
                    Tcompute r_q_norm = rsqrtf(q_norm_sq + 1e-6f);
                    Tcompute r_k_norm = rsqrtf(k_norm_sq + 1e-6f);

                    // Apply normalization
                    for (size_t d = 0; d < Dk; ++d) {
                        q_s[t * Dk + d] *= r_q_norm;
                        k_s[t * Dk + d] *= r_k_norm;
                    }
                }
            }
            __syncthreads();
        }


        // --- 2.3: Intra-Chunk Parallel Scan Logic ---
        // 2.3.1: Perform parallel prefix sum (cumsum) on g_s using CUB
        if (thread_idx < chunk_size) {
            BlockScan(*scan_storage).InclusiveSum(g_s[thread_idx], g_cumsum_s[thread_idx]);
        }
        __syncthreads();

        // 2.3.2: Compute decay_mask & k_beta, v_beta
        for (size_t i = thread_idx; i < chunk_size * chunk_size; i += BLOCK_THREADS) {
            size_t row = i / chunk_size;
            size_t col = i % chunk_size;
            decay_mask_s[i] = (col <= row) ? expf(g_cumsum_s[row] - g_cumsum_s[col]) : 0.0f;
        }
        
        for (size_t i = thread_idx; i < chunk_size; i += BLOCK_THREADS) {
            for (size_t d = 0; d < Dk; ++d) k_s[i * Dk + d] *= (beta_s[i]);
            for (size_t d = 0; d < Dv; ++d) v_s[i * Dv + d] *= beta_s[i];
            for (size_t d = 0; d < Dk; ++d) q_s[i * Dk + d] *= scale;
        }
        __syncthreads();

        // 2.3.3: Compute attn = -((k_beta @ k^T) * decay_mask)
        for (size_t i = thread_idx; i < chunk_size * chunk_size; i += BLOCK_THREADS) {
            size_t row = i / chunk_size;
            size_t col = i % chunk_size;
            Tcompute dot_prod = 0.0f;
            if (col <= row) {
                // Here k_s already contains k_beta
                for(size_t d=0; d<Dk; ++d) dot_prod += k_s[row * Dk + d] * k_s[col * Dk + d];
            }
            attn_s[i] = -dot_prod * decay_mask_s[i];
        }
        __syncthreads();

        // 2.3.4: The complex scan loop for attn matrix
        for (size_t i = 1; i < chunk_size; ++i) {
            for (size_t j = thread_idx; j < i; j += BLOCK_THREADS) {
                Tcompute update_val = 0.0f;
                for (size_t l = 0; l < i; ++l) {
                     update_val += attn_s[i * chunk_size + l] * attn_s[l * chunk_size + j];
                }
                attn_s[i * chunk_size + j] += update_val;
            }
            __syncthreads();
        }
        if (thread_idx < chunk_size) {
            attn_s[thread_idx * chunk_size + thread_idx] += 1.0f;
        }
        __syncthreads();

        // 2.3.5: Compute value_prime = attn @ v_beta and k_cumdecay = attn @ (k_beta * g_exp)
        for (size_t i = thread_idx; i < chunk_size * Dv; i += BLOCK_THREADS) {
            size_t row = i / Dv; size_t col = i % Dv;
            Tcompute dot_prod = 0.0f;
            for(size_t d=0; d<chunk_size; ++d) dot_prod += attn_s[row * chunk_size + d] * v_s[d * Dv + col];
            value_prime_s[i] = dot_prod;
        }
        for (size_t i = thread_idx; i < chunk_size * Dk; i += BLOCK_THREADS) {
            size_t row = i / Dk; int col = i % Dk;
            Tcompute dot_prod = 0.0f;
            for(size_t d=0; d<chunk_size; ++d) dot_prod += attn_s[row * chunk_size + d] * k_s[d * Dk + col] * expf(g_cumsum_s[d]);
            k_cumdecay_s[i] = dot_prod;
        }
        __syncthreads();

        // --- 2.4: Inter-Chunk Interaction ---
        for (size_t i = thread_idx; i < chunk_size * Dv; i += BLOCK_THREADS) {
            size_t row = i / Dv; size_t col = i % Dv;
            Tcompute sum = 0.0f;
            for (size_t d = 0; d < Dk; ++d) sum += k_cumdecay_s[row * Dk + d] * inter_chunk_state_s[d * Dv + col];
            v_prime_s[i] = sum;
            
            sum = 0.0f;
            Tcompute g_exp = expf(g_s[row]);
            for (size_t d = 0; d < Dk; ++d) sum += (q_s[row * Dk + d] * g_exp) * inter_chunk_state_s[d * Dv + col];
            attn_inter_s[i] = sum;
        }
        __syncthreads();

        // --- 2.5: Final Output Calculation and Writeback ---
        for (size_t t = thread_idx; t < chunk_size; t += BLOCK_THREADS) {
            size_t global_t = chunk_offset + t;
            if (global_t < T) {
                // Recompute v_new for this thread
                Tcompute v_new[Dv];
                for(size_t d=0; d<Dv; ++d) v_new[d] = value_prime_s[t * Dv + d] - v_prime_s[t * Dv + d];

                ptrdiff_t out_offset = (batch_idx * T * H * Dv) + (global_t * H * Dv) + (head_idx * Dv);
                for (size_t d_v = 0; d_v < Dv; ++d_v) {
                    Tcompute intra_sum = 0.0f;
                    // Compute (q_t @ k_chunk^T * decay) @ v_new_d_v
                    for (size_t j = 0; j <= t; ++j) { // Causal masking
                        Tcompute dot_qk = 0.0f;
                        for (size_t d_k = 0; d_k < Dk; ++d_k) dot_qk += q_s[t * Dk + d_k] * k_s[j * Dk + d_k]; // k_s is k_beta here
                        
                        // We need to rebuild v_new for index j here
                        Tcompute v_new_j = value_prime_s[j*Dv + d_v] - v_prime_s[j*Dv + d_v];
                        intra_sum += (dot_qk * decay_mask_s[t * chunk_size + j]) * v_new_j;
                    }
                    out[out_offset + d_v] = static_cast<Tdata>(attn_inter_s[t * Dv + d_v] + intra_sum);
                }
            }
        }

        // --- 2.6: Update inter_chunk_state for the next iteration ---
        for (size_t i = thread_idx; i < chunk_size * Dv; i += BLOCK_THREADS) {
            v_new_s[i] = value_prime_s[i] - v_prime_s[i];
        }
        __syncthreads();

        Tcompute g_final_cumsum = g_cumsum_s[chunk_size - 1];
        Tcompute g_final_exp = expf(g_s[chunk_size - 1]); // This might be incorrect, depends on ref logic
                                                          // Let's use the total decay of the chunk
        
        for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
            // First, decay the state with the total decay of the chunk.
            inter_chunk_state_s[i] *= expf(g_final_cumsum - (chunk_idx > 0 ? g_cumsum_s[-1] : 0.0)); // A bit complex
            
            size_t dk = i / Dv; size_t dv = i % Dv;
            Tcompute update_val = 0.0f;
            for (size_t t = 0; t < chunk_size; ++t) {
                Tcompute decay_factor = expf(g_final_cumsum - g_cumsum_s[t]);
                update_val += (k_s[t * Dk + dk] * decay_factor) * v_new_s[t * Dv + dv]; // k_s is k_beta
            }
            atomicAdd(&inter_chunk_state_s[i], update_val);
        }
    }

    // === Phase 3: Write Final State ===
    __syncthreads();
    Tdata* final_state_ptr = final_state + (batch_idx * H + head_idx) * (Dk * Dv);
    for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
        final_state_ptr[i] = static_cast<Tdata>(inter_chunk_state_s[i]);
    }
}

#endif // __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__