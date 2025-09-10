// op/chunk_gated_delta_rule/cuda/kernel.cuh

#ifndef __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__
#define __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__

#include <cuda_fp16.h>
#include <cub/cub.cuh>

// namespace op {
// namespace chunk_gated_delta_rule {
// namespace cuda {

// Define a vectorized type for loading/storing 8 half-precision floats at once.
using half8 = __align__(16) struct { half v[8]; };

template <typename Tdata, typename Tcompute,
          int Dk, int Dv, int CHUNK_SIZE, int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
chunkGatedDeltaRuleKernel(
    Tdata* out,
    Tdata* final_state,
    const Tdata* q,
    const Tdata* k,
    const Tdata* v,
    const Tdata* g,
    const Tdata* beta,
    const Tdata* initial_state,
    bool use_qk_l2norm,
    const int T // Sequence length, passed from host Descriptor._info
) {
    // Grid Strategy: Each block handles one sequence for one head.
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;

    const int T_padded = (T + CHUNK_SIZE - 1) / CHUNK_SIZE * CHUNK_SIZE;
    const int num_chunks = T_padded / CHUNK_SIZE;
    const float scale = rsqrtf(static_cast<Tcompute>(Dk));

    using BlockScan = cub::BlockScan<Tcompute, BLOCK_THREADS>;
    
    // --- Shared Memory Layout ---
    extern __shared__ char shared_mem_char[];
    Tcompute* shared_mem = reinterpret_cast<Tcompute*>(shared_mem_char);
    
    // Memory for one chunk of data
    Tcompute* q_s = shared_mem;
    Tcompute* k_s = q_s + CHUNK_SIZE * Dk;
    Tcompute* v_s = k_s + CHUNK_SIZE * Dk;
    Tcompute* g_s = v_s + CHUNK_SIZE * Dv;
    Tcompute* beta_s = g_s + CHUNK_SIZE;
    
    // Memory for intermediate calculations
    Tcompute* g_cumsum_s = beta_s + CHUNK_SIZE;
    Tcompute* decay_mask_s = g_cumsum_s + CHUNK_SIZE;
    Tcompute* attn_s = decay_mask_s + CHUNK_SIZE * CHUNK_SIZE;
    Tcompute* k_beta_s = attn_s + CHUNK_SIZE * CHUNK_SIZE;
    Tcompute* v_beta_s = k_beta_s + CHUNK_SIZE * Dk;
    Tcompute* k_cumdecay_s = v_beta_s + CHUNK_SIZE * Dv;
    Tcompute* value_prime_s = k_cumdecay_s + CHUNK_SIZE * Dk;
    
    // CUB temporary storage
    typename BlockScan::TempStorage* scan_storage = (typename BlockScan::TempStorage*)(value_prime_s + CHUNK_SIZE * Dv);

    // State passed between chunks
    Tcompute* inter_chunk_state_s = (Tcompute*)(scan_storage + 1);

    // === Phase 1: Initialize inter-chunk state from initial_state ===
    const Tdata* initial_state_ptr = initial_state + (batch_idx * gridDim.y + head_idx) * (Dk * Dv);
    for (int i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
        inter_chunk_state_s[i] = static_cast<Tcompute>(initial_state_ptr[i]);
    }

    // --- Main loop over chunks of the sequence ---
    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        __syncthreads(); // Ensure state is ready for all threads
        int chunk_offset = chunk_idx * CHUNK_SIZE;

        // --- 2.1: Collaborative Loading of chunk data ---
        // Load g and beta
        for (int i = thread_idx; i < CHUNK_SIZE; i += BLOCK_THREADS) {
            int t_idx = chunk_offset + i;
            ptrdiff_t gb_offset = (batch_idx * T + t_idx) * gridDim.y + head_idx;
            g_s[i] = (t_idx < T) ? static_cast<Tcompute>(g[gb_offset]) : 0.0f;
            beta_s[i] = (t_idx < T) ? static_cast<Tcompute>(beta[gb_offset]) : 1.0f; // Use 1 for padding
        }

        // Load q, k, v using vectorized loads
        ptrdiff_t qkv_base_offset = (batch_idx * gridDim.y + head_idx) * T * Dk;
        for (int i = thread_idx; i < CHUNK_SIZE * Dk / 8; i += BLOCK_THREADS) {
            int t_idx = chunk_offset + (i * 8) / Dk;
            if (t_idx < T) {
                ptrdiff_t offset = qkv_base_offset/8 + t_idx * Dk / 8 + (i * 8 % Dk) / 8;
                reinterpret_cast<half8*>(q_s)[i] = reinterpret_cast<const half8*>(q)[offset];
                reinterpret_cast<half8*>(k_s)[i] = reinterpret_cast<const half8*>(k)[offset];
                reinterpret_cast<half8*>(v_s)[i] = reinterpret_cast<const half8*>(v)[offset];
            } else {
                // Zero out padding elements
                reinterpret_cast<float4*>(q_s)[i*2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                reinterpret_cast<float4*>(q_s)[i*2+1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                // Repeat for k_s, v_s
            }
        }
        __syncthreads();

        // --- 2.2: Intra-Chunk Parallel Scan Logic ---
        // 2.2.1: Perform parallel prefix sum (cumsum) on g_s using CUB
        if (thread_idx < CHUNK_SIZE) {
            Tcompute thread_g_val = g_s[thread_idx];
            BlockScan(*scan_storage).InclusiveSum(thread_g_val, g_cumsum_s[thread_idx]);
        }
        __syncthreads();

        // 2.2.2: Compute decay_mask
        for (int i = thread_idx; i < CHUNK_SIZE * CHUNK_SIZE; i += BLOCK_THREADS) {
            int row = i / CHUNK_SIZE;
            int col = i % CHUNK_SIZE;
            decay_mask_s[i] = (col <= row) ? expf(g_cumsum_s[row] - g_cumsum_s[col]) : 0.0f;
        }
        __syncthreads();

        // 2.2.3: Compute k_beta_s and v_beta_s (in-place)
        for (int i = thread_idx; i < CHUNK_SIZE; i += BLOCK_THREADS) {
            for (int d = 0; d < Dk; ++d) k_s[i * Dk + d] *= beta_s[i];
            for (int d = 0; d < Dv; ++d) v_s[i * Dv + d] *= beta_s[i];
        }
        __syncthreads();

        // 2.2.4: Compute attn = -((k_beta @ k^T) * decay_mask)
        for (int i = thread_idx; i < CHUNK_SIZE * CHUNK_SIZE; i += BLOCK_THREADS) {
            int row = i / CHUNK_SIZE;
            int col = i % CHUNK_SIZE;
            Tcompute dot_prod = 0.0f;
            if (col <= row) {
                for(int d=0; d<Dk; ++d) dot_prod += k_s[row * Dk + d] * k_s[col * Dk + d];
            }
            attn_s[i] = -dot_prod * decay_mask_s[i];
        }
        __syncthreads();

        // 2.2.5: The complex scan loop for attn matrix
        for (int i = 1; i < CHUNK_SIZE; ++i) {
            for (int j = thread_idx; j < i; j += BLOCK_THREADS) {
                Tcompute update_val = 0.0f;
                for (int l = 0; l < i; ++l) {
                     update_val += attn_s[i * CHUNK_SIZE + l] * attn_s[l * CHUNK_SIZE + j];
                }
                attn_s[i * CHUNK_SIZE + j] += update_val;
            }
            __syncthreads();
        }

        // Add identity matrix to attn
        if (thread_idx < CHUNK_SIZE) {
            attn_s[thread_idx * CHUNK_SIZE + thread_idx] += 1.0f;
        }
        __syncthreads();

        // 2.2.6: Compute value_prime = attn @ v_beta
        for (int i = thread_idx; i < CHUNK_SIZE * Dv; i += BLOCK_THREADS) {
            int row = i / Dv;
            int col = i % Dv;
            Tcompute dot_prod = 0.0f;
            for(int d=0; d<CHUNK_SIZE; ++d) dot_prod += attn_s[row * CHUNK_SIZE + d] * v_s[d * Dv + col];
            value_prime_s[i] = dot_prod;
        }
        __syncthreads();
        
        // 2.2.7: Compute k_cumdecay = attn @ (k_beta * g_exp)
        // (Similar block-wide matrix multiplication logic)
        
        // --- 2.3: Inter-Chunk State Update and Output Calculation ---
        // (This part involves GEMV-like operations with inter_chunk_state_s)
        
        // --- Write chunk output to global memory ---
        // The result for this chunk is in value_prime_s (and other intermediates).
        // This needs to be combined with the inter-chunk state influence and written out.
        // (Final output calculation and write-back logic)
    }

    // === Phase 4: Write Final State ===
    // Write the final inter_chunk_state_s to global memory
    Tdata* final_state_ptr = final_state + (batch_idx * gridDim.y + head_idx) * (Dk * Dv);
    for (int i = thread_idx; i < Dk * Dv; i += BLOCK_THREADS) {
        final_state_ptr[i] = static_cast<Tdata>(inter_chunk_state_s[i]);
    }
}

// } // namespace cuda
// } // namespace chunk_gated_delta_rule
// } // namespace op

#endif // __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__