// op/chunk_gated_delta_rule/cuda/kernel.cuh

#ifndef __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__
#define __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__

#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <stdio.h>

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
    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_THREADS>;
    
    // --- Shared Memory Layout ---
    extern __shared__ char shared_mem_char[];
    Tcompute* shared_mem = reinterpret_cast<Tcompute*>(shared_mem_char);
    
    // Pointers to different sections of shared memory
    Tcompute* q_s = shared_mem;
    Tcompute* k_s = q_s + chunk_size * Dk;
    Tcompute* v_s = k_s + chunk_size * Dk;
    Tcompute* k_beta_s = v_s + chunk_size * Dv; 
    Tcompute* g_s = k_beta_s + chunk_size * Dk;
    Tcompute* beta_s = g_s + chunk_size;
    Tcompute* g_cumsum_s = beta_s + chunk_size;
    // MODIFIED: Removed decay_mask_s
    Tcompute* attn_s = g_cumsum_s + chunk_size;
    Tcompute* k_cumdecay_s = attn_s + chunk_size * chunk_size;
    Tcompute* value_prime_s = k_cumdecay_s + chunk_size * Dk;
    Tcompute* v_prime_s = value_prime_s + chunk_size * Dv;
    Tcompute* attn_inter_s = v_prime_s + chunk_size * Dv;
    // MODIFIED: Removed v_new_s
    
    typename BlockScan::TempStorage* cub_temp_storage = (typename BlockScan::TempStorage*)(attn_inter_s + chunk_size * Dv);
    typename BlockReduce::TempStorage* reduce_storage = (typename BlockReduce::TempStorage*)(cub_temp_storage + 1);
    Tcompute* inter_chunk_state_s = (Tcompute*)(reduce_storage + 1);

    // === Phase 1: Initialize inter-chunk state from initial_state ===
    if (initial_state != nullptr) {
        // Assuming initial_state layout is [B, H, Dk, Dv]
        const Tdata* initial_state_ptr = initial_state + (batch_idx * H + head_idx) * (Dk * Dv);
        for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
            inter_chunk_state_s[i] = static_cast<Tcompute>(initial_state_ptr[i]);
        }
    } else {
        for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
            inter_chunk_state_s[i] = 0.0f;
        }
    }

    // --- Main loop over chunks of the sequence ---
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        __syncthreads(); // Ensure state is ready and previous chunk's writes are visible
        size_t chunk_offset = chunk_idx * chunk_size;

        // --- 2.1: Collaborative Loading of chunk data ---
        // Load g and beta (Layout: [B, H, T])
        for (size_t i = thread_idx; i < chunk_size; i += BLOCK_THREADS) {
            size_t t_idx = chunk_offset + i;
            if (t_idx < T) {
                ptrdiff_t gb_offset = (batch_idx * H * T) + (head_idx * T) + t_idx;
                g_s[i] = static_cast<Tcompute>(g[gb_offset]);
                beta_s[i] = static_cast<Tcompute>(beta[gb_offset]);
            } else {
                g_s[i] = 0.0f;
                beta_s[i] = 1.0f;
            }
        }

        // Load Q and K (Layout: [B, H, T, Dk])
        for (size_t i = thread_idx; i < chunk_size * Dk; i += BLOCK_THREADS) {
            size_t t_local = i / Dk;
            size_t d = i % Dk;
            size_t t_global = chunk_offset + t_local;
            if (t_global < T) {
                ptrdiff_t qk_offset = (batch_idx * H * T * Dk) + (head_idx * T * Dk) + (t_global * Dk) + d;
                q_s[i] = static_cast<Tcompute>(q[qk_offset]);
                k_s[i] = static_cast<Tcompute>(k[qk_offset]);
            } else {
                q_s[i] = 0.0f;
                k_s[i] = 0.0f;
            }
        }

        // Load V (Layout: [B, H, T, Dv])
        for (size_t i = thread_idx; i < chunk_size * Dv; i += BLOCK_THREADS) {
             size_t t_local = i / Dv;
             size_t d = i % Dv;
             size_t t_global = chunk_offset + t_local;
             if (t_global < T) {
                ptrdiff_t v_offset = (batch_idx * H * T * Dv) + (head_idx * T * Dv) + (t_global * Dv) + d;
                v_s[i] = static_cast<Tcompute>(v[v_offset]);
             } else {
                v_s[i] = 0.0f;
             }
        }
        __syncthreads();

        // --- 2.2: Optional L2 Normalization ---
        if (use_qk_l2norm) {
            for (size_t t = thread_idx; t < chunk_size; t += BLOCK_THREADS) {
                size_t t_global = chunk_offset + t;
                if (t_global < T) {
                    Tcompute q_norm_sq = 0.0f;
                    Tcompute k_norm_sq = 0.0f;
                    for (size_t d = 0; d < Dk; ++d) {
                        Tcompute q_val = q_s[t * Dk + d];
                        Tcompute k_val = k_s[t * Dk + d];
                        q_norm_sq += q_val * q_val;
                        k_norm_sq += k_val * k_val;
                    }
                    Tcompute r_q_norm = rsqrtf(q_norm_sq + 1e-6f);
                    Tcompute r_k_norm = rsqrtf(k_norm_sq + 1e-6f);
                    for (size_t d = 0; d < Dk; ++d) {
                        q_s[t * Dk + d] *= r_q_norm;
                        k_s[t * Dk + d] *= r_k_norm;
                    }
                }
            }
            __syncthreads();
        }

        // 2.3.1: Perform parallel prefix sum (cumsum) on g_s using CUB
        Tcompute g_val = (thread_idx < chunk_size) ? g_s[thread_idx] : 0.0f;
        Tcompute g_cumsum_val;
        BlockScan(*cub_temp_storage).InclusiveSum(g_val, g_cumsum_val);
        if (thread_idx < chunk_size) {
            g_cumsum_s[thread_idx] = g_cumsum_val;
        }
        __syncthreads(); 

        // 2.3.2: Compute k_beta, v_beta, and scaled q
        // MODIFIED: Removed decay_mask_s calculation loop
        for (size_t i = thread_idx; i < chunk_size; i += BLOCK_THREADS) {
            Tcompute beta_val = beta_s[i];
            for (size_t d = 0; d < Dk; ++d) k_beta_s[i * Dk + d] = k_s[i * Dk + d] * beta_val;
            for (size_t d = 0; d < Dv; ++d) v_s[i * Dv + d] *= beta_val; // v_s becomes v_beta
            for (size_t d = 0; d < Dk; ++d) q_s[i * Dk + d] *= scale; // q_s becomes q_scaled
        }
        __syncthreads();

        // 2.3.3: Compute attn = -((k_beta @ k^T) * decay_mask)
        for (size_t i = thread_idx; i < chunk_size * chunk_size; i += BLOCK_THREADS) {
            size_t row = i / chunk_size;
            size_t col = i % chunk_size;
            Tcompute dot_prod = 0.0f;
            if (col < row) {
                for(size_t d = 0; d < Dk; ++d) {
                    dot_prod += k_beta_s[row * Dk + d] * k_s[col * Dk + d];
                }
                // MODIFIED: On-the-fly calculation of decay_mask value
                Tcompute decay = expf(g_cumsum_s[row] - g_cumsum_s[col]);
                attn_s[i] = -dot_prod * decay;
            } else {
                attn_s[i] = 0.0f;
            }
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
            size_t row = i / Dv; size_t col_v = i % Dv;
            Tcompute dot_prod = 0.0f;
            for(size_t d=0; d<chunk_size; ++d) dot_prod += attn_s[row * chunk_size + d] * v_s[d * Dv + col_v];
            value_prime_s[i] = dot_prod;
        }
        
        for (size_t i = thread_idx; i < chunk_size * Dk; i += BLOCK_THREADS) {
            size_t row = i / Dk; int col_k = i % Dk;
            Tcompute dot_prod = 0.0f;
            for(size_t d=0; d<chunk_size; ++d) dot_prod += attn_s[row * chunk_size + d] * k_beta_s[d * Dk + col_k] * expf(g_cumsum_s[d]);
            k_cumdecay_s[i] = dot_prod;
        }
        __syncthreads();

        // --- 2.4: Inter-Chunk Interaction ---
        // Calculate v_prime
        for (size_t i = thread_idx; i < chunk_size * Dv; i += BLOCK_THREADS) {
            size_t row = i / Dv; size_t col_v = i % Dv;
            Tcompute sum = 0.0f;
            for (size_t d = 0; d < Dk; ++d) sum += k_cumdecay_s[row * Dk + d] * inter_chunk_state_s[d * Dv + col_v];
            v_prime_s[i] = sum;
        }
        // Calculate attn_inter
        for (size_t i = thread_idx; i < chunk_size * Dv; i += BLOCK_THREADS) {
            size_t row = i / Dv; size_t col_v = i % Dv;
            Tcompute sum = 0.0f;
            Tcompute g_exp = expf(g_cumsum_s[row]);
            for (size_t d = 0; d < Dk; ++d) sum += (q_s[row * Dk + d] * g_exp) * inter_chunk_state_s[d * Dv + col_v];
            attn_inter_s[i] = sum;
        }
        __syncthreads();

        // MODIFIED: Removed the explicit calculation and storage of v_new_s

        // --- 2.5: Final Output Calculation and Writeback ---
        for (size_t t = thread_idx; t < chunk_size; t += BLOCK_THREADS) {
            size_t global_t = chunk_offset + t;
            if (global_t < T) {
                ptrdiff_t out_offset = (batch_idx * H * T * Dv) + (head_idx * T * Dv) + (global_t * Dv);
                for (size_t d_v = 0; d_v < Dv; ++d_v) {
                    Tcompute intra_sum = 0.0f;
                    for (size_t j = 0; j <= t; ++j) { // Causal masking
                        Tcompute dot_qk = 0.0f;
                        for (size_t d_k = 0; d_k < Dk; ++d_k) dot_qk += q_s[t * Dk + d_k] * k_s[j * Dk + d_k];
                        
                        // MODIFIED: On-the-fly calculation of v_new and decay_mask value
                        Tcompute value_prime_j = value_prime_s[j * Dv + d_v];
                        Tcompute v_prime_j = v_prime_s[j * Dv + d_v];
                        Tcompute v_new_j = value_prime_j - v_prime_j;
                        Tcompute decay = expf(g_cumsum_s[t] - g_cumsum_s[j]);

                        intra_sum += (dot_qk * decay) * v_new_j;
                    }
                    out[out_offset + d_v] = static_cast<Tdata>(attn_inter_s[t * Dv + d_v] + intra_sum);
                }
            }
        }

        // --- 2.6: Update inter_chunk_state for the next iteration ---
        Tcompute g_final_cumsum = g_cumsum_s[chunk_size - 1];
        
        // Step 1: Decay the old state
        for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
            inter_chunk_state_s[i] *= expf(g_final_cumsum);
        }
        __syncthreads();

        // Step 2: Add the contribution from the current chunk
        for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
            size_t dk = i / Dv; size_t dv = i % Dv;
            Tcompute update_val = 0.0f;
            for (size_t t = 0; t < chunk_size; ++t) {
                Tcompute decay_factor = expf(g_final_cumsum - g_cumsum_s[t]);
                
                // MODIFIED: On-the-fly calculation of v_new value
                Tcompute value_prime_t = value_prime_s[t * Dv + dv];
                Tcompute v_prime_t = v_prime_s[t * Dv + dv];
                Tcompute v_new_t = value_prime_t - v_prime_t;

                update_val += (k_s[t * Dk + dk] * decay_factor) * v_new_t;
            }
            atomicAdd(&inter_chunk_state_s[i], update_val);
        }
    }

    // === Phase 3: Write Final State ===
    __syncthreads();
    // Assuming final_state layout is [B, H, Dk, Dv]
    Tdata* final_state_ptr = final_state + (batch_idx * H + head_idx) * (Dk * Dv);
    for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
        final_state_ptr[i] = static_cast<Tdata>(inter_chunk_state_s[i]);
    }
}

#endif // __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__