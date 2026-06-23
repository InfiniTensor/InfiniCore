// kernel.cuh (in op/recurrent_gated_delta_rule/cuda/)

#ifndef __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__
#define __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ inline int64_t loadStateIndex(
    const void *indices,
    bool is_i64,
    int batch_idx,
    int fallback) {
    if (indices == nullptr) {
        return static_cast<int64_t>(fallback);
    }
    return is_i64
             ? static_cast<const int64_t *>(indices)[batch_idx]
             : static_cast<int64_t>(static_cast<const int32_t *>(indices)[batch_idx]);
}

template <typename T>
__device__ inline float loadAsFloat(const T *ptr, ptrdiff_t offset) {
    return static_cast<float>(ptr[offset]);
}

template <>
__device__ inline float loadAsFloat<half>(const half *ptr, ptrdiff_t offset) {
    return __half2float(ptr[offset]);
}

template <>
__device__ inline float loadAsFloat<__nv_bfloat16>(const __nv_bfloat16 *ptr, ptrdiff_t offset) {
    return __bfloat162float(ptr[offset]);
}

template <typename Tdata, typename Tgate, typename Tcompute, size_t Dk, size_t Dv, size_t NUM_THREADS>
__device__ void recurrentGatedDeltaRuleKernel(
    Tdata *out,
    Tdata *initial_state,
    Tdata *final_state,
    const Tdata *q,
    const Tdata *k,
    const Tdata *v,
    const Tgate *g,
    const Tgate *beta,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool use_qk_l2norm,
    bool indexed_state_pool,
    size_t Hk,
    size_t value_heads_per_key_head,
    ptrdiff_t out_s0,
    ptrdiff_t out_s1,
    ptrdiff_t out_s2,
    ptrdiff_t initial_s0,
    ptrdiff_t initial_s1,
    ptrdiff_t initial_s2,
    ptrdiff_t initial_s3,
    ptrdiff_t final_s0,
    ptrdiff_t final_s1,
    ptrdiff_t final_s2,
    ptrdiff_t final_s3,
    ptrdiff_t q_s0,
    ptrdiff_t q_s1,
    ptrdiff_t q_s2,
    ptrdiff_t k_s0,
    ptrdiff_t k_s1,
    ptrdiff_t k_s2,
    ptrdiff_t v_s0,
    ptrdiff_t v_s1,
    ptrdiff_t v_s2,
    ptrdiff_t g_s0,
    ptrdiff_t g_s1,
    ptrdiff_t g_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s1,
    ptrdiff_t beta_s2) {
    const int batch_idx = blockIdx.x;
    const int value_head_idx = blockIdx.y;
    const int key_head_idx = value_head_idx / static_cast<int>(value_heads_per_key_head);
    const int thread_idx = threadIdx.x;

    if (key_head_idx >= static_cast<int>(Hk)) {
        return;
    }

    constexpr int seq_idx = 0;
    const ptrdiff_t q_base = static_cast<ptrdiff_t>(batch_idx) * q_s0 + seq_idx * q_s1 + static_cast<ptrdiff_t>(key_head_idx) * q_s2;
    const ptrdiff_t k_base = static_cast<ptrdiff_t>(batch_idx) * k_s0 + seq_idx * k_s1 + static_cast<ptrdiff_t>(key_head_idx) * k_s2;
    const ptrdiff_t v_base = static_cast<ptrdiff_t>(batch_idx) * v_s0 + seq_idx * v_s1 + static_cast<ptrdiff_t>(value_head_idx) * v_s2;
    const ptrdiff_t out_base = static_cast<ptrdiff_t>(batch_idx) * out_s0 + seq_idx * out_s1 + static_cast<ptrdiff_t>(value_head_idx) * out_s2;
    const ptrdiff_t gate_offset = static_cast<ptrdiff_t>(batch_idx) * g_s0 + seq_idx * g_s1 + static_cast<ptrdiff_t>(value_head_idx) * g_s2;
    const ptrdiff_t beta_offset = static_cast<ptrdiff_t>(batch_idx) * beta_s0 + seq_idx * beta_s1 + static_cast<ptrdiff_t>(value_head_idx) * beta_s2;

    int64_t read_slot = static_cast<int64_t>(batch_idx);
    int64_t write_slot = static_cast<int64_t>(batch_idx);
    if (indexed_state_pool) {
        read_slot = loadStateIndex(initial_state_indices, initial_state_indices_i64, batch_idx, batch_idx);
        write_slot = final_state_indices == nullptr
                       ? static_cast<int64_t>(batch_idx)
                       : loadStateIndex(final_state_indices, final_state_indices_i64, batch_idx, batch_idx);
        if (read_slot < 0 || write_slot < 0) {
            for (int dv_idx = thread_idx; dv_idx < Dv; dv_idx += NUM_THREADS) {
                out[out_base + dv_idx] = static_cast<Tdata>(0.0f);
            }
            return;
        }
    }

    const ptrdiff_t initial_base = indexed_state_pool
                                     ? static_cast<ptrdiff_t>(read_slot) * initial_s0 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1
                                     : static_cast<ptrdiff_t>(batch_idx) * initial_s0 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1;
    ptrdiff_t final_base = 0;
    Tdata *final_state_target = nullptr;
    if (indexed_state_pool && final_state_indices != nullptr) {
        final_state_target = initial_state;
        final_base = static_cast<ptrdiff_t>(write_slot) * initial_s0 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1;
    } else if (indexed_state_pool) {
        final_state_target = final_state;
        final_base = static_cast<ptrdiff_t>(batch_idx) * final_s0 + static_cast<ptrdiff_t>(value_head_idx) * final_s1;
    } else {
        final_state_target = final_state;
        final_base = static_cast<ptrdiff_t>(batch_idx) * final_s0 + static_cast<ptrdiff_t>(value_head_idx) * final_s1;
    }

    extern __shared__ char shared_mem_char[];
    Tcompute *shared_mem = reinterpret_cast<Tcompute *>(shared_mem_char);

    Tcompute *q_local = shared_mem;
    Tcompute *k_local = q_local + Dk;
    Tcompute *norm_val = k_local + Dk;

    for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
        q_local[i] = static_cast<Tcompute>(loadAsFloat(q, q_base + i));
        k_local[i] = static_cast<Tcompute>(loadAsFloat(k, k_base + i));
    }

    if (use_qk_l2norm) {
        __syncthreads();
        Tcompute sum_sq = 0.0f;
        for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
            sum_sq += q_local[i] * q_local[i];
        }
        norm_val[thread_idx] = sum_sq;
        __syncthreads();
        if (thread_idx == 0) {
            Tcompute total_sum_sq = 0.0f;
            for (int i = 0; i < NUM_THREADS; ++i) {
                total_sum_sq += norm_val[i];
            }
            norm_val[0] = rsqrtf(total_sum_sq + 1e-6f);
        }
        __syncthreads();
        Tcompute r_norm_q = norm_val[0];

        for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
            q_local[i] *= r_norm_q;
        }

        sum_sq = 0.0f;
        for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
            sum_sq += k_local[i] * k_local[i];
        }
        norm_val[thread_idx] = sum_sq;
        __syncthreads();
        if (thread_idx == 0) {
            Tcompute total_sum_sq = 0.0f;
            for (int i = 0; i < NUM_THREADS; ++i) {
                total_sum_sq += norm_val[i];
            }
            norm_val[0] = rsqrtf(total_sum_sq + 1e-6f);
        }
        __syncthreads();
        Tcompute r_norm_k = norm_val[0];

        for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
            k_local[i] *= r_norm_k;
        }
        __syncthreads();
    }

    Tcompute g_t = expf(static_cast<Tcompute>(loadAsFloat(g, gate_offset)));
    Tcompute beta_t = static_cast<Tcompute>(loadAsFloat(beta, beta_offset));
    Tcompute scale = rsqrtf(static_cast<Tcompute>(Dk));

    for (int i = thread_idx; i < Dk; i += NUM_THREADS) {
        q_local[i] *= scale;
    }
    __syncthreads();

    for (int dv_idx = thread_idx; dv_idx < Dv; dv_idx += NUM_THREADS) {
        Tcompute kv_mem = 0.0f;
        for (int dk_idx = 0; dk_idx < Dk; ++dk_idx) {
            ptrdiff_t state_idx = indexed_state_pool
                                    ? initial_base + static_cast<ptrdiff_t>(dv_idx) * initial_s2 + static_cast<ptrdiff_t>(dk_idx) * initial_s3
                                    : initial_base + static_cast<ptrdiff_t>(dk_idx) * initial_s2 + static_cast<ptrdiff_t>(dv_idx) * initial_s3;
            Tcompute h_prev = static_cast<Tcompute>(loadAsFloat(initial_state, state_idx));
            kv_mem += (h_prev * g_t) * k_local[dk_idx];
        }

        Tcompute v_t = static_cast<Tcompute>(loadAsFloat(v, v_base + dv_idx));
        Tcompute delta = (v_t - kv_mem) * beta_t;
        Tcompute out_val = 0.0f;

        for (int dk_idx = 0; dk_idx < Dk; ++dk_idx) {
            ptrdiff_t read_state_idx = indexed_state_pool
                                         ? initial_base + static_cast<ptrdiff_t>(dv_idx) * initial_s2 + static_cast<ptrdiff_t>(dk_idx) * initial_s3
                                         : initial_base + static_cast<ptrdiff_t>(dk_idx) * initial_s2 + static_cast<ptrdiff_t>(dv_idx) * initial_s3;
            ptrdiff_t write_state_idx = indexed_state_pool
                                          ? final_base + static_cast<ptrdiff_t>(dv_idx) * (final_state_indices != nullptr ? initial_s2 : final_s2) + static_cast<ptrdiff_t>(dk_idx) * (final_state_indices != nullptr ? initial_s3 : final_s3)
                                          : final_base + static_cast<ptrdiff_t>(dk_idx) * final_s2 + static_cast<ptrdiff_t>(dv_idx) * final_s3;
            Tcompute h_prev = static_cast<Tcompute>(loadAsFloat(initial_state, read_state_idx));
            Tcompute h_final = (h_prev * g_t) + (k_local[dk_idx] * delta);
            out_val += h_final * q_local[dk_idx];
            final_state_target[write_state_idx] = static_cast<Tdata>(h_final);
        }
        out[out_base + dv_idx] = static_cast<Tdata>(out_val);
    }
}

#endif // __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__
