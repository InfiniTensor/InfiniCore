#ifndef __FLASH_ATTENTION_KERNEL_CUH__
#define __FLASH_ATTENTION_KERNEL_CUH__

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace flash_attention_cuda {

// Constants for Flash Attention algorithm
constexpr int BLOCK_SIZE_M = 32; // Number of Q rows per block
constexpr int BLOCK_SIZE_N = 32; // Number of K/V columns per block
constexpr int BLOCK_SIZE_D = 64; // Head dimension block size

template <typename T>
__device__ __forceinline__ T gelu(T x) {
    return x; // Placeholder, implement actual GELU if needed
}

template <typename T, typename Tcompute>
__device__ __forceinline__ Tcompute warp_reduce_sum(Tcompute val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename Tdata, typename Tcompute>
__global__ void flash_attention_forward_kernel(
    Tdata *__restrict__ out,
    const Tdata *__restrict__ q,
    const Tdata *__restrict__ k,
    const Tdata *__restrict__ v,
    const int64_t *__restrict__ total_kv_len,
    size_t batch_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t seq_len_q,
    size_t seq_len_kv,
    size_t head_dim,
    float scale,
    char is_causal,
    char has_variable_kv_len,
    ptrdiff_t q_stride_batch,
    ptrdiff_t q_stride_head,
    ptrdiff_t q_stride_seq,
    ptrdiff_t k_stride_batch,
    ptrdiff_t k_stride_head,
    ptrdiff_t k_stride_seq,
    ptrdiff_t v_stride_batch,
    ptrdiff_t v_stride_head,
    ptrdiff_t v_stride_seq,
    ptrdiff_t out_stride_batch,
    ptrdiff_t out_stride_head,
    ptrdiff_t out_stride_seq,
    float *__restrict__ softmax_lse) {

    // Each block handles a (batch, head, block of Q rows)
    size_t batch_id = blockIdx.y;
    size_t head_id = blockIdx.z;
    size_t kv_head_id = num_kv_heads == num_heads ? head_id : head_id / (num_heads / num_kv_heads);

    size_t q_start_row = blockIdx.x * BLOCK_SIZE_M;
    size_t q_end_row = min(q_start_row + BLOCK_SIZE_M, seq_len_q);

    // Adjust KV sequence length if variable
    int effective_kv_len = has_variable_kv_len ? total_kv_len[batch_id] : seq_len_kv;

    // Pointers to current batch/head
    const Tdata *q_batch_head = q + batch_id * q_stride_batch + head_id * q_stride_head;
    const Tdata *k_batch_head = k + batch_id * k_stride_batch + kv_head_id * k_stride_head;
    const Tdata *v_batch_head = v + batch_id * v_stride_batch + kv_head_id * v_stride_head;
    Tdata *out_batch_head = out + batch_id * out_stride_batch + head_id * out_stride_head;

    // Shared memory for Q block
    __shared__ Tdata q_block[BLOCK_SIZE_M][BLOCK_SIZE_D];

// Load Q block
#pragma unroll
    for (int i = threadIdx.y; i < BLOCK_SIZE_M && (q_start_row + i) < q_end_row; i += blockDim.y) {
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            if (j < head_dim) {
                q_block[i][j] = q_batch_head[(q_start_row + i) * q_stride_seq + j];
            }
        }
    }
    __syncthreads();

    // Initialize accumulators and statistics
    Tcompute acc[BLOCK_SIZE_M][BLOCK_SIZE_D] = {{0}};
    Tcompute m[BLOCK_SIZE_M] = {-INFINITY};
    Tcompute l[BLOCK_SIZE_M] = {0};

    // Iterate over KV blocks
    for (int kv_start_col = 0; kv_start_col < effective_kv_len; kv_start_col += BLOCK_SIZE_N) {
        int kv_end_col = min(kv_start_col + BLOCK_SIZE_N, effective_kv_len);

        // Shared memory for K/V block
        __shared__ Tdata k_block[BLOCK_SIZE_N][BLOCK_SIZE_D];
        __shared__ Tdata v_block[BLOCK_SIZE_N][BLOCK_SIZE_D];

// Load K block
#pragma unroll
        for (int i = threadIdx.y; i < BLOCK_SIZE_N && (kv_start_col + i) < kv_end_col; i += blockDim.y) {
            for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
                if (j < head_dim) {
                    k_block[i][j] = k_batch_head[(kv_start_col + i) * k_stride_seq + j];
                }
            }
        }

// Load V block
#pragma unroll
        for (int i = threadIdx.y; i < BLOCK_SIZE_N && (kv_start_col + i) < kv_end_col; i += blockDim.y) {
            for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
                if (j < head_dim) {
                    v_block[i][j] = v_batch_head[(kv_start_col + i) * v_stride_seq + j];
                }
            }
        }
        __syncthreads();

        // Compute Q*K^T for this block
        for (int i = 0; i < BLOCK_SIZE_M && (q_start_row + i) < q_end_row; i++) {
            for (int j = 0; j < BLOCK_SIZE_N && (kv_start_col + j) < kv_end_col; j++) {
                // Check causality
                if (is_causal && (q_start_row + i) < (kv_start_col + j)) {
                    continue;
                }

                // Compute dot product
                Tcompute score = 0;
#pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    score += Tcompute(q_block[i][d]) * Tcompute(k_block[j][d]);
                }
                score *= Tcompute(scale);

                // Online softmax update
                Tcompute m_ij = max(m[i], score);
                Tcompute exp_diff = exp(m[i] - m_ij);
                Tcompute exp_score = exp(score - m_ij);

// Update accumulators
#pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    acc[i][d] = acc[i][d] * exp_diff + exp_score * Tcompute(v_block[j][d]);
                }

                l[i] = l[i] * exp_diff + exp_score;
                m[i] = m_ij;
            }
        }
        __syncthreads();
    }

// Write output
#pragma unroll
    for (int i = threadIdx.y; i < BLOCK_SIZE_M && (q_start_row + i) < q_end_row; i += blockDim.y) {
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            if (j < head_dim) {
                out_batch_head[(q_start_row + i) * out_stride_seq + j] = Tdata(acc[i][j] / l[i]);
            }
        }
    }

    // Store softmax LSE if needed (for backward pass)
    if (softmax_lse != nullptr && threadIdx.x == 0 && threadIdx.y == 0) {
        size_t idx = (batch_id * num_heads + head_id) * seq_len_q + q_start_row;
        for (int i = 0; i < BLOCK_SIZE_M && (q_start_row + i) < q_end_row; i++) {
            softmax_lse[idx + i] = m[i] + log(l[i]);
        }
    }
}

// Main launcher function
template <typename Tdata, typename Tcompute>
infiniStatus_t flash_attention_forward(
    Tdata *out,
    const Tdata *q,
    const Tdata *k,
    const Tdata *v,
    const int64_t *total_kv_len,
    size_t batch_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t seq_len_q,
    size_t seq_len_kv,
    size_t head_dim,
    float scale,
    char is_causal,
    char has_variable_kv_len,
    ptrdiff_t q_stride_batch,
    ptrdiff_t q_stride_head,
    ptrdiff_t q_stride_seq,
    ptrdiff_t k_stride_batch,
    ptrdiff_t k_stride_head,
    ptrdiff_t k_stride_seq,
    ptrdiff_t v_stride_batch,
    ptrdiff_t v_stride_head,
    ptrdiff_t v_stride_seq,
    ptrdiff_t out_stride_batch,
    ptrdiff_t out_stride_head,
    ptrdiff_t out_stride_seq,
    float *softmax_lse,
    size_t softmax_lse_size,
    cudaStream_t stream) {

    dim3 block_dim(32, 4); // 128 threads total
    dim3 grid_dim(
        (seq_len_q + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
        batch_size,
        num_heads);

    flash_attention_forward_kernel<Tdata, Tcompute>
        <<<grid_dim, block_dim, 0, stream>>>(
            out, q, k, v, total_kv_len,
            batch_size, num_heads, num_kv_heads,
            seq_len_q, seq_len_kv, head_dim,
            scale, is_causal, has_variable_kv_len,
            q_stride_batch, q_stride_head, q_stride_seq,
            k_stride_batch, k_stride_head, k_stride_seq,
            v_stride_batch, v_stride_head, v_stride_seq,
            out_stride_batch, out_stride_head, out_stride_seq,
            softmax_lse);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace flash_attention_cuda

#endif // __FLASH_ATTENTION_KERNEL_CUH__
