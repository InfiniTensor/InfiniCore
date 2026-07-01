#ifndef __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__
#define __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__

#include <cmath>
#include <cstdint>

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
__device__ inline float loadAsFloat<cuda_bfloat16>(const cuda_bfloat16 *ptr, ptrdiff_t offset) {
    return __bfloat162float(ptr[offset]);
}

__device__ inline float warpReduceSum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}
template <typename Tdata, typename Tgate, typename Tcompute, size_t Dk, size_t Dv, size_t WARPS_PER_BLOCK>
__device__ void recurrentGatedDeltaRuleIndexedPoolWarpKernel(
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
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_THREADS = WARPS_PER_BLOCK * WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int value_head_idx = blockIdx.y;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x & (WARP_SIZE - 1);
    const int value_dim_idx = blockIdx.z * WARPS_PER_BLOCK + warp_idx;
    const int key_head_idx = value_head_idx / static_cast<int>(value_heads_per_key_head);

    if (key_head_idx >= static_cast<int>(Hk)) {
        return;
    }

    constexpr int seq_idx = 0;
    const ptrdiff_t q_base = static_cast<ptrdiff_t>(batch_idx) * q_s0 + seq_idx * q_s1 + static_cast<ptrdiff_t>(key_head_idx) * q_s2;
    const ptrdiff_t k_base = static_cast<ptrdiff_t>(batch_idx) * k_s0 + seq_idx * k_s1 + static_cast<ptrdiff_t>(key_head_idx) * k_s2;

    extern __shared__ char shared_mem_char[];
    Tcompute *shared_mem = reinterpret_cast<Tcompute *>(shared_mem_char);
    Tcompute *q_local = shared_mem;
    Tcompute *k_local = q_local + Dk;
    Tcompute *norm_val = k_local + Dk;

    for (int i = threadIdx.x; i < static_cast<int>(Dk); i += NUM_THREADS) {
        q_local[i] = static_cast<Tcompute>(loadAsFloat(q, q_base + i));
        k_local[i] = static_cast<Tcompute>(loadAsFloat(k, k_base + i));
    }

    if (use_qk_l2norm) {
        __syncthreads();
        Tcompute sum_sq = 0.0f;
        for (int i = threadIdx.x; i < static_cast<int>(Dk); i += NUM_THREADS) {
            sum_sq += q_local[i] * q_local[i];
        }
        norm_val[threadIdx.x] = sum_sq;
        __syncthreads();
        if (threadIdx.x == 0) {
            Tcompute total_sum_sq = 0.0f;
            for (int i = 0; i < NUM_THREADS; ++i) {
                total_sum_sq += norm_val[i];
            }
            norm_val[0] = rsqrtf(total_sum_sq + 1e-6f);
        }
        __syncthreads();
        const Tcompute r_norm_q = norm_val[0];

        for (int i = threadIdx.x; i < static_cast<int>(Dk); i += NUM_THREADS) {
            q_local[i] *= r_norm_q;
        }

        sum_sq = 0.0f;
        for (int i = threadIdx.x; i < static_cast<int>(Dk); i += NUM_THREADS) {
            sum_sq += k_local[i] * k_local[i];
        }
        norm_val[threadIdx.x] = sum_sq;
        __syncthreads();
        if (threadIdx.x == 0) {
            Tcompute total_sum_sq = 0.0f;
            for (int i = 0; i < NUM_THREADS; ++i) {
                total_sum_sq += norm_val[i];
            }
            norm_val[0] = rsqrtf(total_sum_sq + 1e-6f);
        }
        __syncthreads();
        const Tcompute r_norm_k = norm_val[0];

        for (int i = threadIdx.x; i < static_cast<int>(Dk); i += NUM_THREADS) {
            k_local[i] *= r_norm_k;
        }
    }

    const Tcompute scale = rsqrtf(static_cast<Tcompute>(Dk));
    for (int i = threadIdx.x; i < static_cast<int>(Dk); i += NUM_THREADS) {
        q_local[i] *= scale;
    }
    __syncthreads();

    if (value_dim_idx >= static_cast<int>(Dv)) {
        return;
    }

    int64_t read_slot = loadStateIndex(initial_state_indices, initial_state_indices_i64, batch_idx, batch_idx);
    int64_t write_slot = final_state_indices == nullptr
                           ? static_cast<int64_t>(batch_idx)
                           : loadStateIndex(final_state_indices, final_state_indices_i64, batch_idx, batch_idx);

    const ptrdiff_t out_base = static_cast<ptrdiff_t>(batch_idx) * out_s0 + seq_idx * out_s1 + static_cast<ptrdiff_t>(value_head_idx) * out_s2;
    if (read_slot < 0 || write_slot < 0) {
        if (lane_idx == 0) {
            out[out_base + value_dim_idx] = static_cast<Tdata>(0.0f);
        }
        return;
    }

    const ptrdiff_t v_base = static_cast<ptrdiff_t>(batch_idx) * v_s0 + seq_idx * v_s1 + static_cast<ptrdiff_t>(value_head_idx) * v_s2;
    const ptrdiff_t gate_offset = static_cast<ptrdiff_t>(batch_idx) * g_s0 + seq_idx * g_s1 + static_cast<ptrdiff_t>(value_head_idx) * g_s2;
    const ptrdiff_t beta_offset = static_cast<ptrdiff_t>(batch_idx) * beta_s0 + seq_idx * beta_s1 + static_cast<ptrdiff_t>(value_head_idx) * beta_s2;

    const ptrdiff_t initial_base = static_cast<ptrdiff_t>(read_slot) * initial_s0
                                 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1
                                 + static_cast<ptrdiff_t>(value_dim_idx) * initial_s2;

    Tdata *final_state_target = final_state_indices == nullptr ? final_state : initial_state;
    const ptrdiff_t final_base = final_state_indices == nullptr
                                   ? static_cast<ptrdiff_t>(batch_idx) * final_s0
                                         + static_cast<ptrdiff_t>(value_head_idx) * final_s1
                                         + static_cast<ptrdiff_t>(value_dim_idx) * final_s2
                                   : static_cast<ptrdiff_t>(write_slot) * initial_s0
                                         + static_cast<ptrdiff_t>(value_head_idx) * initial_s1
                                         + static_cast<ptrdiff_t>(value_dim_idx) * initial_s2;
    const ptrdiff_t final_k_stride = final_state_indices == nullptr ? final_s3 : initial_s3;

    const Tcompute g_t = expf(static_cast<Tcompute>(loadAsFloat(g, gate_offset)));
    const Tcompute beta_t = static_cast<Tcompute>(loadAsFloat(beta, beta_offset));

    Tcompute kv_mem = 0.0f;
    Tcompute hq_mem = 0.0f;
    Tcompute kq_mem = 0.0f;
    for (int dk_idx = lane_idx; dk_idx < static_cast<int>(Dk); dk_idx += WARP_SIZE) {
        const Tcompute h_prev = static_cast<Tcompute>(loadAsFloat(initial_state, initial_base + static_cast<ptrdiff_t>(dk_idx) * initial_s3));
        const Tcompute k_t = k_local[dk_idx];
        const Tcompute q_t = q_local[dk_idx];
        kv_mem += (h_prev * g_t) * k_t;
        hq_mem += h_prev * q_t;
        kq_mem += k_t * q_t;
    }
    kv_mem = warpReduceSum(kv_mem);
    hq_mem = warpReduceSum(hq_mem);
    kq_mem = warpReduceSum(kq_mem);

    const Tcompute v_t = static_cast<Tcompute>(loadAsFloat(v, v_base + value_dim_idx));
    const Tcompute delta = (v_t - kv_mem) * beta_t;

    if (lane_idx == 0) {
        const Tcompute out_val = g_t * hq_mem + delta * kq_mem;
        out[out_base + value_dim_idx] = static_cast<Tdata>(out_val);
    }

    for (int dk_idx = lane_idx; dk_idx < static_cast<int>(Dk); dk_idx += WARP_SIZE) {
        const Tcompute h_prev = static_cast<Tcompute>(loadAsFloat(initial_state, initial_base + static_cast<ptrdiff_t>(dk_idx) * initial_s3));
        const Tcompute h_final = (h_prev * g_t) + (k_local[dk_idx] * delta);
        final_state_target[final_base + static_cast<ptrdiff_t>(dk_idx) * final_k_stride] = static_cast<Tdata>(h_final);
    }
}
#endif // __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__
