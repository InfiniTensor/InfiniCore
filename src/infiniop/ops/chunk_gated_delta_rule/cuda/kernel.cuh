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
    Tcompute* decay_mask_s = g_cumsum_s + chunk_size;
    Tcompute* attn_s = decay_mask_s + chunk_size * chunk_size;
    Tcompute* k_cumdecay_s = attn_s + chunk_size * chunk_size;
    Tcompute* value_prime_s = k_cumdecay_s + chunk_size * Dk;
    Tcompute* v_prime_s = value_prime_s + chunk_size * Dv;
    Tcompute* attn_inter_s = v_prime_s + chunk_size * Dv;
    // MODIFIED: Added a dedicated shared memory array for v_new
    Tcompute* v_new_s = attn_inter_s + chunk_size * Dv;
    
    // typename BlockScan::TempStorage* scan_storage = (typename BlockScan::TempStorage*)(v_new_s + chunk_size * Dv);
    // Tcompute* inter_chunk_state_s = (Tcompute*)(scan_storage + 1);
    // --- MODIFICATION START: Renamed for clarity, can be used by both Scan and Reduce ---
    typename BlockScan::TempStorage* cub_temp_storage = (typename BlockScan::TempStorage*)(v_new_s + chunk_size * Dv);
    typename BlockReduce::TempStorage* reduce_storage = (typename BlockReduce::TempStorage*)(cub_temp_storage + 1);
    // --- MODIFICATION END ---
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
    // // ==================== DEBUG PRINT 2a =====================
    //     __syncthreads();
    //     if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0) {
    //         printf("--- CUDA Kernel: initial_state (= (dot_qk * decay_mask_s[t * chunk_size + j]) * v_new_j;) ---\n");
    //         for (size_t i = 0; i < Dk; ++i) {
    //             for (size_t j = 0; j < Dv; ++j) {
    //             // for (size_t j = Dv-chunk_size; j < Dv; ++j) {
    //                 printf("%8.4f, %llu ", (float)initial_state[i * Dk + j], (unsigned long long)(i * Dk + j));
    //             }
    //             printf("\n");
    //         }
    //         printf("------------------------------------------\n");
    //     }
    //     __syncthreads();
    // // =========================================================


    // --- Main loop over chunks of the sequence ---
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        __syncthreads(); // Ensure state is ready and previous chunk's writes are visible
        size_t chunk_offset = chunk_idx * chunk_size;



        // --- 2.1: Collaborative Loading of chunk data (with updated indexing) ---
        // --- MODIFICATION START: Updated memory indexing for [B, H, T, D] layout ---
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

        
        // __syncthreads();
        // // ==================== DEBUG PRINT 2a =====================
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
        //     printf("--- CUDA Kernel: g_s (BEFORE scan) ---\n");
        //     for (size_t j = 0; j < chunk_size; ++j) {
        //         printf("%8.4f ", (float)g_s[j]);
        //     }
        //     printf("------------------------------------------\n");
        // }
        // __syncthreads();
        // // =========================================================

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
        
        // __syncthreads();
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
        //     printf("--- CUDA Kernel: attn_s (BEFORE scan) ---\n");
        //     for (size_t i = 0; i < chunk_size; ++i) {
        //         for (size_t j = 0; j < chunk_size; ++j) {
        //             printf("%8.4f ", (float)k_s[i * chunk_size + j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("------------------------------------------\n");
        // }
        // __syncthreads();



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
        // --- MODIFICATION END ---
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


        // 2.3.2: Compute decay_mask, k_beta, v_beta, and scaled q
        for (size_t i = thread_idx; i < chunk_size * chunk_size; i += BLOCK_THREADS) {
            size_t row = i / chunk_size;
            size_t col = i % chunk_size;
            if (col <= row) {
                decay_mask_s[i] = expf(g_cumsum_s[row] - g_cumsum_s[col]);
            } else {
                decay_mask_s[i] = 0.0f; // Also zero out the upper triangle
            }
        }
        
        for (size_t i = thread_idx; i < chunk_size; i += BLOCK_THREADS) {
            Tcompute beta_val = beta_s[i];
            // --- START MODIFICATION 2 ---
            // Calculate k_beta_s, leaving original k_s untouched
            for (size_t d = 0; d < Dk; ++d) k_beta_s[i * Dk + d] = k_s[i * Dk + d] * beta_val;
            // --- END MODIFICATION 2 ---
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
                // --- START MODIFICATION 2 ---
                // Correctly compute dot product of k_beta (row) and k (col)
                for(size_t d = 0; d < Dk; ++d) {
                    dot_prod += k_beta_s[row * Dk + d] * k_s[col * Dk + d];
                }
                attn_s[i] = -dot_prod * decay_mask_s[i];
                // --- END MODIFICATION 2 ---
            } else {
                attn_s[i] = 0.0f; // Zero out the upper triangle
            }
        }
        __syncthreads();

        


        // // ==================== DEBUG PRINT 2a =====================
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
        //     printf("--- CUDA Kernel: attn_s (BEFORE scan) ---\n");
        //     for (size_t i = 0; i < chunk_size; ++i) {
        //         for (size_t j = 0; j < chunk_size; ++j) {
        //             printf("%8.4f ", (float)attn_s[i * chunk_size + j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("------------------------------------------\n");
        // }
        // __syncthreads();
        // // =========================================================
        
        // // ==================== DEBUG PRINT 2a =====================
        // __syncthreads();
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
        //     printf("--- CUDA Kernel: attn_s (before Tcompute beta_val = beta_s[i];) ---\n");
        //     for (size_t i = 0; i < chunk_size; ++i) {
        //         for (size_t j = 0; j < chunk_size; ++j) {
        //             printf("%8.4f ", (float)attn_s[i * chunk_size + j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("------------------------------------------\n");
        // }
        // __syncthreads();
        // // =========================================================


        // 2.3.4: The complex scan loop for attn matrix
        // NOTE: The following sequential loop remains a performance bottleneck due to the algorithm's nature.
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
        //         // ==================== DEBUG PRINT 2b =====================
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
        //     printf("--- CUDA Kernel: attn_s (AFTER scan) ---\n");
        //     for (size_t i = 0; i < chunk_size; ++i) {
        //         for (size_t j = 0; j < chunk_size; ++j) {
        //             printf("%8.4f ", (float)attn_s[i * chunk_size + j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("------------------------------------------\n");
        // }
        // // =========================================================
        // __syncthreads();
        
        // printf("check ok.\n");
        // ==================== DEBUG PRINT 2a =====================
        __syncthreads();
        if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 1) {
            printf("--- CUDA Kernel: attn_s (Tcompute beta_val = beta_s[i];) ---\n");
            for (size_t i = 0; i < chunk_size; ++i) {
                for (size_t j = 0; j < chunk_size; ++j) {
                    printf("%8.4f ", (float)attn_s[i * chunk_size + j]);
                }
                printf("\n");
            }
            printf("------------------------------------------\n");
        }
        __syncthreads();
        // =========================================================

        if (thread_idx < chunk_size) {
            attn_s[thread_idx * chunk_size + thread_idx] += 1.0f;
        }
        __syncthreads();

        // 2.3.5: Compute value = attn @ v_beta and k_cumdecay = attn @ (k_beta * g_exp)
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

        // ==================== DEBUG PRINT 2a =====================
        __syncthreads();
        if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 1) {
            printf("--- CUDA Kernel: value_prime_s (Tcompute beta_val = beta_s[i];) ---\n");
            for (size_t i = 0; i < chunk_size; ++i) {
                for (size_t j = 0; j < chunk_size; ++j) {
                    printf("%8.4f ", (float)value_prime_s[i * Dv + j]);
                }
                printf("\n");
            }
            printf("------------------------------------------\n");
        }
        __syncthreads();
        // =========================================================

        // // ==================== DEBUG PRINT 2a =====================
        // __syncthreads();
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
        //     printf("--- CUDA Kernel: k_cumdecay_s (Tcompute beta_val = beta_s[i];) ---\n");
        //     for (size_t i = 0; i < chunk_size; ++i) {
        //         for (size_t j = 0; j < chunk_size; ++j) {
        //             printf("%8.4f ", (float)k_cumdecay_s[i * chunk_size + j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("------------------------------------------\n");
        // }
        // __syncthreads();
        // // =========================================================

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

        // FIX 1: Correctly calculate v_new = value - v_prime
        for (size_t i = thread_idx; i < chunk_size * Dv; i += BLOCK_THREADS) {
            v_new_s[i] = value_prime_s[i] - v_prime_s[i];
        }
        __syncthreads();


        // // ==================== DEBUG PRINT 3 =====================
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
        //     printf("--- CUDA Kernel: v_new_s (chunk 0) ---\n");
        //     // Print the first 8 tokens and their first 4 dimensions
        //     for (size_t t = 0; t < 8; ++t) {
        //         printf("t=%zu: ", t);
        //         for (size_t d = 0; d < 4; ++d) {
        //             printf("%8.4f ", (float)v_new_s[t * Dv + d]);
        //         }
        //         printf("\n");
        //     }
        //     printf("------------------------------------------\n");
        // }
        // // ======================================================

        // // ==================== DEBUG PRINT 2a =====================
        // __syncthreads();
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 1) {
        //     printf("--- CUDA Kernel: v_new_s (= (dot_qk * decay_mask_s[t * chunk_size + j]) * v_new_j;) ---\n");
        //     for (size_t i = 0; i < chunk_size; ++i) {
        //         // for (size_t j = 0; j < chunk_size; ++j) {
        //         for (size_t j = Dv-chunk_size; j < Dv; ++j) {
        //             printf("%8.4f, %llu ", (float)v_new_s[i * Dv + j], (unsigned long long)(i * Dv + j));
        //         }
        //         printf("\n");
        //     }
        //     printf("------------------------------------------\n");
        // }
        // __syncthreads();
        // // =========================================================
        // // ==================== DEBUG PRINT 2a =====================
        // __syncthreads();
        // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 1) {
        //     printf("--- CUDA Kernel: attn_inter_s (= (dot_qk * decay_mask_s[t * chunk_size + j]) * v_new_j;) ---\n");
        //     for (size_t i = 0; i < chunk_size; ++i) {
        //         // for (size_t j = 0; j < chunk_size; ++j) {
        //         for (size_t j = Dv-chunk_size; j < Dv; ++j) {
        //             printf("%8.4f, %llu ", (float)attn_inter_s[i * Dv + j], (unsigned long long)(i * Dv + j));
        //         }
        //         printf("\n");
        //     }
        //     printf("------------------------------------------\n");
        // }
        // __syncthreads();
        // // =========================================================
        // ==================== DEBUG PRINT 2a =====================
        __syncthreads();
        if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 1) {
            printf("--- CUDA Kernel: v_prime_s (= (dot_qk * decay_mask_s[t * chunk_size + j]) * v_new_j;) ---\n");
            for (size_t i = 0; i < chunk_size; ++i) {
                // for (size_t j = 0; j < chunk_size; ++j) {
                for (size_t j = Dv-chunk_size; j < Dv; ++j) {
                    printf("%8.4f, %llu ", (float)v_prime_s[i * Dv + j], (unsigned long long)(i * Dv + j));
                }
                printf("\n");
            }
            printf("------------------------------------------\n");
        }
        __syncthreads();
        // =========================================================+





        // ==================== DEBUG PRINT 2a =====================
        __syncthreads();
        if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 1) {
            printf("--- CUDA Kernel: attn_inter_s final ---\n");
            for (size_t i = 0; i < chunk_size; ++i) {
                // for (size_t j = 0; j < chunk_size; ++j) {
                for (size_t j = 0; j < chunk_size; ++j) {
                    printf("%8.4f, %llu ", (float)attn_inter_s[i * Dv + j], (unsigned long long)(i * Dv + j));
                }
                printf("\n");
            }
            printf("------------------------------------------\n");
        }
        __syncthreads();
        // =========================================================

        // --- 2.5: Final Output Calculation and Writeback ---
        // NOTE: This section correctly implements the reference algorithm, but the algorithm
        // itself involves redundant computation (re-calculating simple attention).
        for (size_t t = thread_idx; t < chunk_size; t += BLOCK_THREADS) {
            size_t global_t = chunk_offset + t;
            if (global_t < T) {
                // --- MODIFICATION START: Updated memory indexing for output ---
                ptrdiff_t out_offset = (batch_idx * H * T * Dv) + (head_idx * T * Dv) + (global_t * Dv);
                Tdata output_it;
                // --- MODIFICATION END ---
                for (size_t d_v = 0; d_v < Dv; ++d_v) {
                    Tcompute intra_sum = 0.0f;
                    for (size_t j = 0; j <= t; ++j) { // Causal masking
                        Tcompute dot_qk = 0.0f;
                        for (size_t d_k = 0; d_k < Dk; ++d_k) dot_qk += q_s[t * Dk + d_k] * k_s[j * Dk + d_k];
                        Tcompute v_new_j = v_new_s[j * Dv + d_v];
                        intra_sum += (dot_qk * decay_mask_s[t * chunk_size + j]) * v_new_j;
                    }
                    
                    
                    out[out_offset + d_v] = static_cast<Tdata>(attn_inter_s[t * Dv + d_v] + intra_sum);
                    // output_it = static_cast<Tdata>(attn_inter_s[t * Dv + d_v] + intra_sum);
                    // if (batch_idx == 0 && head_idx == 0 && d_v == 0){
                    //     printf("intra_sum: %8.4f, attn_inter_s: %8.4f, idx: %llu, t: %llu\n", (float)intra_sum, (float)attn_inter_s[t * Dv + d_v], 
                    //     (unsigned long long)(t * Dv + d_v), (unsigned long long)(t));
                    // }
                    // out[out_offset + d_v] = output_it;
                }
            }
        }

        // --- 2.6: Update inter_chunk_state for the next iteration ---
        Tcompute g_final_cumsum = g_cumsum_s[chunk_size - 1];
        
        // // Step 1: Decay the old state
        // for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
        //     inter_chunk_state_s[i] *= expf(g_final_cumsum);
        // }
        // __syncthreads();

        

        // // --- MODIFICATION START: Replaced atomicAdd with efficient parallel reduction ---
        // // Step 2: Compute and add the contribution from the current chunk.
        // // We loop through each element of the state matrix S. For each element, the entire
        // // thread block collaborates to compute the update sum in parallel.
        // for (size_t i = 0; i < Dk * Dv; ++i) {
        //     size_t dk = i / Dv; 
        //     size_t dv = i % Dv;
            
        //     // Step 2.1: Each thread computes its partial sum over the chunk dimension 't'.
        //     Tcompute partial_sum = 0.0f;
        //     for (size_t t = thread_idx; t < chunk_size; t += BLOCK_THREADS) {
        //         Tcompute decay_factor = expf(g_final_cumsum - g_cumsum_s[t]);
        //         partial_sum += (k_beta_s[t * Dk + dk] * decay_factor) * v_new_s[t * Dv + dv];
        //     }
            
        //     // Step 2.2: The block reduces all partial sums into a single total sum.
        //     Tcompute total_update_sum;
        //     BlockReduce(*reduce_storage).Sum(partial_sum, total_update_sum);


        //     // Step 2.3: A single thread adds the final sum to the state. This avoids atomic conflicts.
        //     if (thread_idx == 0) {
        //         inter_chunk_state_s[i] += total_update_sum;
        //     }


        //     // Step 2.4: Sync after each state element update to ensure the next reduction is correct.
        //     // This is necessary because CUB collectives use shared memory implicitly.
        //     __syncthreads();
        // }
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
                // --- START MODIFICATION 3 ---
                // Use k_beta_s for the update, not k_s
                update_val += (k_s[t * Dk + dk] * decay_factor) * v_new_s[t * Dv + dv];
                // --- END MODIFICATION 3 ---
            }
            atomicAdd(&inter_chunk_state_s[i], update_val);
        }

        
        // ==================== DEBUG PRINT 2a =====================
        __syncthreads();
        if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
            printf("--- CUDA Kernel:  decay_factor) * v_new_s chunk_idx0 (= (tast_recurrent_state * decay_mask_s[t * chunk_size + j]) * v_new_j;) ---\n");
            for (size_t i = 0; i < chunk_size; ++i) {
                for (size_t j = 0; j < chunk_size; ++j) {
                // for (size_t j = Dv-chunk_size; j < Dv; ++j) {
                    printf("%8.4f, %llu ", (float)inter_chunk_state_s[i * Dv + j], (unsigned long long)(i * Dv + j));
                }
                printf("\n");
            }
            printf("------------------------------------------\n");
        }
        __syncthreads();
        // =========================================================

        
            
        // ==================== DEBUG PRINT 2a =====================
        __syncthreads();
        if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 && chunk_idx == 0) {
            printf("--- CUDA Kernel: inter_chunk_state_s chunk_idx0 (= (dot_qk * decay_mask_s[t * chunk_size + j]) * v_new_j;) ---\n");
            for (size_t i = 0; i < chunk_size; ++i) {
                for (size_t j = 0; j < chunk_size; ++j) {
                // for (size_t j = Dv-chunk_size; j < Dv; ++j) {
                    printf("%8.4f, %llu ", (float)inter_chunk_state_s[i * Dv + j], (unsigned long long)(i * Dv + j));
                }
                printf("\n");
            }
            printf("------------------------------------------\n");
        }
        __syncthreads();
        // =========================================================
    }
    // // ==================== DEBUG PRINT 4 =====================
    // __syncthreads();
    // if (batch_idx == 0 && head_idx == 0 && threadIdx.x == 0 ) {
    //     printf("--- CUDA Kernel: inter_chunk_state_s (chunk 0) ---\n");
    //     // Print the first 8 tokens and their first 4 dimensions
    //     for (size_t t = 0; t < 8; ++t) {
    //         printf("t=%zu: ", t);
    //         for (size_t d = 0; d < 4; ++d) {
    //             printf("%8.4f ", (float)inter_chunk_state_s[t * Dv + d]);
    //         }
    //         printf("\n");
    //     }
    //     printf("------------------------------------------\n");
    // }
    // // ======================================================

    

    // === Phase 3: Write Final State ===
    __syncthreads();
    // Assuming final_state layout is [B, H, Dk, Dv]
    Tdata* final_state_ptr = final_state + (batch_idx * H + head_idx) * (Dk * Dv);
    for (size_t i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
        final_state_ptr[i] = static_cast<Tdata>(inter_chunk_state_s[i]);
    }
}

#endif // __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__