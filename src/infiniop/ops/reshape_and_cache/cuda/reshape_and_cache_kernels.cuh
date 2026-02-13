#pragma once

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../paged_attention_v2/utils/dtype_fp8.cuh"
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <float.h>

namespace op::reshape_and_cache::cuda {

using Fp8KVCacheDataType = op::paged_attention_v2::vllm::Fp8KVCacheDataType;

// Used to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
    float scale;

    __device__ __forceinline__ void operator()(OutT &dst, const InT src) const {
        if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
            dst = static_cast<OutT>(src);
        } else {
            // dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);
            assert(false);
        }
    }
};

// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
    scalar_t val[vec_size];
};

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
    ScaOp scalar_op;

    __device__ __forceinline__ void operator()(
        vec_n_t<OutT, VEC_SIZE> &dst, const vec_n_t<InT, VEC_SIZE> &src) const {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            scalar_op(dst.val[i], src.val[i]);
        }
    }
};

template <int VEC_SIZE, typename InT, typename OutT, typename VecOp,
          typename ScaOp>
__device__ inline void vectorize_with_alignment(
    const InT *in, OutT *out, int len, int tid, int stride,
    VecOp &&vec_op,      // vec_n_t<InT,16> -> vec_n_t<OutT,16>
    ScaOp &&scalar_op) { // InT -> OutT
    static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                  "VEC_SIZE must be a positive power-of-two");
    constexpr int WIDTH = VEC_SIZE * sizeof(InT); // eg: 64 B
    uintptr_t addr = reinterpret_cast<uintptr_t>(in);

    // fast path when the whole region is already aligned
    // Note: currently the output is guaranteed to be same as the input, so we
    // don't check it here, comments here just for future reference.
    bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
    if (can_vec) {
        int num_vec = len / VEC_SIZE;

        using vin_t = vec_n_t<InT, VEC_SIZE>;
        using vout_t = vec_n_t<OutT, VEC_SIZE>;
        auto *v_in = reinterpret_cast<const vin_t *>(in);
        auto *v_out = reinterpret_cast<vout_t *>(out);

        for (int i = tid; i < num_vec; i += stride) {
            vout_t tmp;
            // Make a local copy of the entire pack
            vin_t src = v_in[i]; // <- encourages a single vector ld
            vec_op(tmp, src);
            v_out[i] = tmp; // <- encourages a single vector st
        }
        return;
    }

    int misalignment_offset = addr & (WIDTH - 1);      // addr % 64
    int alignment_bytes = WIDTH - misalignment_offset; // 64 - (addr % 64)
    int prefix_elems = alignment_bytes & (WIDTH - 1);  // handle 64
    prefix_elems /= sizeof(InT);
    prefix_elems = min(prefix_elems, len); // 0 ≤ prefix < 16

    // 1. prefill the when it is unsafe to vectorize
    for (int i = tid; i < prefix_elems; i += stride) {
        scalar_op(out[i], in[i]);
    }

    in += prefix_elems;
    out += prefix_elems;
    len -= prefix_elems;

    int num_vec = len / VEC_SIZE;
    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    auto *v_in = reinterpret_cast<const vin_t *>(in);
    auto *v_out = reinterpret_cast<vout_t *>(out);

    // 2. vectorize the main part
    for (int i = tid; i < num_vec; i += stride) {
        vout_t tmp;
        // Make a local copy of the entire pack
        vin_t src = v_in[i]; // <- encourages a single vector ld
        vec_op(tmp, src);
        v_out[i] = tmp; // <- encourages a single vector st
    }

    // 3. handle the tail
    int tail_start = num_vec * VEC_SIZE;
    for (int i = tid + tail_start; i < len; i += stride) {
        scalar_op(out[i], in[i]);
    }
}

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
__device__ __forceinline__ void vectorize_with_alignment(const InT *in,
                                                         OutT *out, int len,
                                                         int tid, int stride,
                                                         ScaOp &&scalar_op) {
    using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>>;
    vectorize_with_alignment<VEC_SIZE>(in, out, len, tid, stride, Vec{scalar_op},
                                       std::forward<ScaOp>(scalar_op));
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t *__restrict__ key,         // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value,       // [num_tokens, num_heads, head_size]
    cache_t *__restrict__ key_cache,          // [num_blocks, num_heads, head_size/x,
                                              // block_size, x]
    cache_t *__restrict__ value_cache,        // [num_blocks, num_heads, head_size,
                                              // block_size]
    const int64_t *__restrict__ slot_mapping, // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float *k_scale, const float *v_scale) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int h_block_count = head_size / x; // head_size//x

    const int h_block_idx = threadIdx.x;
    if (h_block_idx >= num_heads * h_block_count) {
        return;
    }

    const int head_idx = h_block_idx / h_block_count;
    const int h_block = h_block_idx % h_block_count;

    const scalar_t *__restrict__ key_src = key + token_idx * key_stride + head_idx * head_size + h_block * x;
    const int64_t src_value_start = token_idx * value_stride + head_idx * head_size + h_block * x;

    cache_t *__restrict__ key_dst = key_cache + block_idx * num_heads * h_block_count * block_size * x + head_idx * h_block_count * block_size * x + h_block * block_size * x + block_offset * x;
    const int64_t tgt_value_start = block_idx * num_heads * h_block_count * x * block_size + head_idx * h_block_count * x * block_size + h_block * x * block_size + block_offset;

    constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
    float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
    float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

    vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, x, 0, 1, k_op);

    const scalar_t *__restrict__ value_src = value + src_value_start;
    cache_t *__restrict__ value_dst = value_cache + tgt_value_start;
#pragma unroll
    for (int i = 0; i < x; i++) {
        v_op(value_dst[i * block_size], value_src[i]);
    }
}
} // namespace op::reshape_and_cache::cuda
