// Decode-path (small M) GEMV for GGUF block-quantized weights, NVIDIA backend.
//
//   C[M, N] = A[M, K] @ W[N, K]^T
//
// A is BF16 row-major.  C is BF16 on the normal path, with an opt-in F32 output
// variant for strict-consistency diagnostics.  W is the packed GGUF blob: N rows
// of `row_bytes` bytes, each row holding K / block_elems(TYPE) quantizer blocks
// of the given ggml type.  Nothing is ever expanded to BF16 in memory -- a block
// is decoded into registers and immediately consumed by the dot product, which
// is the whole point of route B (weights stay at ~6.56 bits per element and the
// kernel is bandwidth bound on W).
//
// Block decoding is *not* duplicated here: it comes from ggml_blocks.h, the same
// single source of truth that was verified bit-exactly against numpy / gguf-py
// for both host and device in stage 3.1.
//
// Layout of the work: one warp owns one output row n, lane l takes quantizer
// blocks l, l+32, ... along K, and each lane accumulates a private fp32 partial
// per input row.  A warp-local shuffle reduction then writes C.  That keeps the
// weight row contiguous per warp (the access pattern that matters, W being ~99%
// of the traffic) and needs no shared memory for M <= kMaxDecodeM.
//
// Status: correctness-first v1 (stage 3.2).  The reduction order is "sum inside
// a block, then across blocks per lane, then across lanes", so results differ
// from a plain fp32 matmul in the last ulp only; the acceptance gate is
// cos_sim > 0.999 against the dense reference (plan 1.2 item 2).  Vectorized
// weight loads and a shared-memory A tile are stage 6 work.
#ifndef __LINEAR_GGUF_NVIDIA_GEMV_CUH__
#define __LINEAR_GGUF_NVIDIA_GEMV_CUH__

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../ggml_blocks.h"

namespace op::linear_gguf::nvidia {

// Compile-time register capacity.  The shipped dispatch still uses M <= 8 by
// default; an opt-in strict-consistency experiment may use the extra slots for
// short prefills (M <= 16) without changing the normal path.
constexpr int kMaxDecodeM = 16;

template <int32_t TYPE>
struct BlockTraits;

template <>
struct BlockTraits<ggml_blocks::GGML_TYPE_Q8_0> {
    static constexpr int32_t kBytes = ggml_blocks::SIZE_Q8_0;
    static constexpr int32_t kElems = ggml_blocks::QK8_0;
};
template <>
struct BlockTraits<ggml_blocks::GGML_TYPE_Q4_K> {
    static constexpr int32_t kBytes = ggml_blocks::SIZE_Q4_K;
    static constexpr int32_t kElems = ggml_blocks::QK_K;
};
template <>
struct BlockTraits<ggml_blocks::GGML_TYPE_Q5_K> {
    static constexpr int32_t kBytes = ggml_blocks::SIZE_Q5_K;
    static constexpr int32_t kElems = ggml_blocks::QK_K;
};
template <>
struct BlockTraits<ggml_blocks::GGML_TYPE_Q6_K> {
    static constexpr int32_t kBytes = ggml_blocks::SIZE_Q6_K;
    static constexpr int32_t kElems = ggml_blocks::QK_K;
};

// Forwards to the shared decoders; TYPE keeps the branch compile-time so only
// one call survives instantiation.
template <int32_t TYPE>
__device__ __forceinline__ void decode_block(const uint8_t *blk, float *out) {
    if constexpr (TYPE == ggml_blocks::GGML_TYPE_Q8_0) {
        ggml_blocks::decode_q8_0(blk, out);
    } else if constexpr (TYPE == ggml_blocks::GGML_TYPE_Q4_K) {
        ggml_blocks::decode_q4_K(blk, out);
    } else if constexpr (TYPE == ggml_blocks::GGML_TYPE_Q5_K) {
        ggml_blocks::decode_q5_K(blk, out);
    } else {
        ggml_blocks::decode_q6_K(blk, out);
    }
}

__device__ __forceinline__ void store_gemv_value(__nv_bfloat16 *c, int64_t idx, float v) {
    c[idx] = __float2bfloat16_rn(v);
}

__device__ __forceinline__ void store_gemv_value(float *c, int64_t idx, float v) {
    c[idx] = v;
}

// Return the signed 6-bit integer code for element j of one Q6_K super-block.
// The layout follows block_q6_K in llama.cpp: two 128-element halves, with
// low nibbles in ql and two high bits in qh.
__device__ __forceinline__ int q6_k_code(const uint8_t *blk, int j) {
    const int half = j >> 7;
    const int within = j & 127;
    const int quarter = within >> 5;
    const int lane = within & 31;
    const uint8_t *ql = blk + half * 64;
    const uint8_t *qh = blk + ggml_blocks::Q6K_OFF_QH + half * 32;
    int low;
    if (quarter == 0) {
        low = ql[lane] & 0x0f;
    } else if (quarter == 1) {
        low = ql[lane + 32] & 0x0f;
    } else if (quarter == 2) {
        low = ql[lane] >> 4;
    } else {
        low = ql[lane + 32] >> 4;
    }
    const int high = (qh[lane] >> (2 * quarter)) & 0x03;
    return (low | (high << 4)) - 32;
}

template <int32_t TYPE, bool QUANTIZE_ACTIVATION, typename OutT>
__global__ void gemv_decode_kernel(const __nv_bfloat16 *__restrict__ a,
                                   const uint8_t *__restrict__ w,
                                   OutT *__restrict__ c, int m_count, int n_count,
                                   int k, int64_t row_bytes) {
    constexpr int32_t kBytes = BlockTraits<TYPE>::kBytes;
    constexpr int32_t kElems = BlockTraits<TYPE>::kElems;

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
    const int n = blockIdx.x * warps_per_block + warp;
    if (n >= n_count) return;

    const uint8_t *wrow = w + static_cast<int64_t>(n) * row_bytes;
    const int blocks_per_row = k / kElems;

    float acc[kMaxDecodeM];
#pragma unroll
    for (int m = 0; m < kMaxDecodeM; ++m) acc[m] = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        float wk[kElems];
        decode_block<TYPE>(wrow + static_cast<int64_t>(b) * kBytes, wk);
        const __nv_bfloat16 *ab = a + static_cast<int64_t>(b) * kElems;
#pragma unroll
        for (int m = 0; m < kMaxDecodeM; ++m) {
            if (m >= m_count) break;
            const __nv_bfloat16 *am = ab + static_cast<int64_t>(m) * k;
            float s = 0.0f;
            if constexpr (QUANTIZE_ACTIVATION) {
                // llama.cpp's CUDA matvec path quantizes the activation side to
                // Q8_1 in groups of 32 before the quantized vec-dot.
                if constexpr (TYPE == ggml_blocks::GGML_TYPE_Q6_K) {
                    // Formula-level Q6_K x Q8_1 path: preserve integer q6/q8
                    // dot products and sub-block scales instead of multiplying
                    // two independently dequantized float vectors.
                    const uint8_t *blk = wrow + static_cast<int64_t>(b) * kBytes;
                    const int8_t *scales = reinterpret_cast<const int8_t *>(
                        blk + ggml_blocks::Q6K_OFF_SCALES);
                    const float d6 = ggml_blocks::half_to_float(
                        ggml_blocks::read_u16(blk + ggml_blocks::Q6K_OFF_D));
                    float scaled_integer_sum = 0.0f;
#pragma unroll
                    for (int base = 0; base < kElems; base += 32) {
                        float amax = 0.0f;
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            amax = fmaxf(amax, fabsf(__bfloat162float(__ldg(am + base + j))));
                        }
                        const float d_inv = 127.0f / amax;
                        const float d8 = 1.0f / d_inv;
                        int dot0 = 0;
                        int dot1 = 0;
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const float av = __bfloat162float(__ldg(am + base + j));
                            const int q8 = amax == 0.0f ? 0 : static_cast<int>(roundf(av * d_inv));
                            dot0 += q8 * q6_k_code(blk, base + j);
                        }
#pragma unroll
                        for (int j = 16; j < 32; ++j) {
                            const float av = __bfloat162float(__ldg(am + base + j));
                            const int q8 = amax == 0.0f ? 0 : static_cast<int>(roundf(av * d_inv));
                            dot1 += q8 * q6_k_code(blk, base + j);
                        }
                        const int scale_idx = base / 16;
                        const int scaled_dot =
                            static_cast<int>(scales[scale_idx]) * dot0
                            + static_cast<int>(scales[scale_idx + 1]) * dot1;
                        scaled_integer_sum += d8 * static_cast<float>(scaled_dot);
                    }
                    s = d6 * scaled_integer_sum;
                } else {
#pragma unroll
                    for (int base = 0; base < kElems; base += 32) {
                        float amax = 0.0f;
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            amax = fmaxf(amax, fabsf(__bfloat162float(__ldg(am + base + j))));
                        }
                        const float d_inv = 127.0f / amax;
                        const float d = 1.0f / d_inv;
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            const float av = __bfloat162float(__ldg(am + base + j));
                            const float q = amax == 0.0f ? 0.0f : roundf(av * d_inv);
                            s += (d * q) * wk[base + j];
                        }
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < kElems; ++j) {
                    s += __bfloat162float(__ldg(am + j)) * wk[j];
                }
            }
            acc[m] += s;
        }
    }

    const unsigned members = 0xffffffffu;
#pragma unroll
    for (int m = 0; m < kMaxDecodeM; ++m) {
        if (m >= m_count) break;
        float v = acc[m];
#pragma unroll
        for (int off = 16; off; off >>= 1) {
            v += __shfl_down_sync(members, v, off);
        }
        if (lane == 0) {
            store_gemv_value(c, static_cast<int64_t>(m) * n_count + n, v);
        }
    }
}

constexpr int kGemvThreads = 256;

// Returns false when `type` has no decoder here or the geometry is not a whole
// number of blocks per row, without launching anything.  The caller turns that
// into an error: silently falling back to a dense weight would defeat the point
// of route B.
template <typename OutT>
inline bool launch_gemv_decode_typed(int32_t type, const __nv_bfloat16 *a, const uint8_t *w,
                                     OutT *c, int m_count, int n_count, int k,
                                     int64_t row_bytes, bool quantize_activation,
                                     cudaStream_t stream) {
    if (m_count <= 0 || m_count > kMaxDecodeM || n_count <= 0 || k <= 0) return false;
    const int32_t elems = ggml_blocks::block_elems(type);
    const int32_t bytes = ggml_blocks::block_bytes(type);
    if (elems <= 0 || bytes <= 0) return false;
    if (k % elems != 0) return false;
    if (row_bytes < static_cast<int64_t>(k / elems) * bytes) return false;

    const int warps_per_block = kGemvThreads >> 5;
    const unsigned grid = static_cast<unsigned>((n_count + warps_per_block - 1) / warps_per_block);
    if (quantize_activation) {
        switch (type) {
        case ggml_blocks::GGML_TYPE_Q8_0:
            gemv_decode_kernel<ggml_blocks::GGML_TYPE_Q8_0, true, OutT><<<grid, kGemvThreads, 0, stream>>>(
                a, w, c, m_count, n_count, k, row_bytes);
            break;
        case ggml_blocks::GGML_TYPE_Q4_K:
            gemv_decode_kernel<ggml_blocks::GGML_TYPE_Q4_K, true, OutT><<<grid, kGemvThreads, 0, stream>>>(
                a, w, c, m_count, n_count, k, row_bytes);
            break;
        case ggml_blocks::GGML_TYPE_Q5_K:
            gemv_decode_kernel<ggml_blocks::GGML_TYPE_Q5_K, true, OutT><<<grid, kGemvThreads, 0, stream>>>(
                a, w, c, m_count, n_count, k, row_bytes);
            break;
        case ggml_blocks::GGML_TYPE_Q6_K:
            gemv_decode_kernel<ggml_blocks::GGML_TYPE_Q6_K, true, OutT><<<grid, kGemvThreads, 0, stream>>>(
                a, w, c, m_count, n_count, k, row_bytes);
            break;
        default:
            return false;
        }
        return cudaGetLastError() == cudaSuccess;
    }

    switch (type) {
    case ggml_blocks::GGML_TYPE_Q8_0:
        gemv_decode_kernel<ggml_blocks::GGML_TYPE_Q8_0, false, OutT><<<grid, kGemvThreads, 0, stream>>>(
            a, w, c, m_count, n_count, k, row_bytes);
        break;
    case ggml_blocks::GGML_TYPE_Q4_K:
        gemv_decode_kernel<ggml_blocks::GGML_TYPE_Q4_K, false, OutT><<<grid, kGemvThreads, 0, stream>>>(
            a, w, c, m_count, n_count, k, row_bytes);
        break;
    case ggml_blocks::GGML_TYPE_Q5_K:
        gemv_decode_kernel<ggml_blocks::GGML_TYPE_Q5_K, false, OutT><<<grid, kGemvThreads, 0, stream>>>(
            a, w, c, m_count, n_count, k, row_bytes);
        break;
    case ggml_blocks::GGML_TYPE_Q6_K:
        gemv_decode_kernel<ggml_blocks::GGML_TYPE_Q6_K, false, OutT><<<grid, kGemvThreads, 0, stream>>>(
            a, w, c, m_count, n_count, k, row_bytes);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

inline bool launch_gemv_decode(int32_t type, const __nv_bfloat16 *a, const uint8_t *w,
                               __nv_bfloat16 *c, int m_count, int n_count, int k,
                               int64_t row_bytes, bool quantize_activation,
                               cudaStream_t stream) {
    return launch_gemv_decode_typed(type, a, w, c, m_count, n_count, k,
                                    row_bytes, quantize_activation, stream);
}

inline bool launch_gemv_decode_f32(int32_t type, const __nv_bfloat16 *a, const uint8_t *w,
                                   float *c, int m_count, int n_count, int k,
                                   int64_t row_bytes, bool quantize_activation,
                                   cudaStream_t stream) {
    return launch_gemv_decode_typed(type, a, w, c, m_count, n_count, k,
                                    row_bytes, quantize_activation, stream);
}

}  // namespace op::linear_gguf::nvidia

#endif  // __LINEAR_GGUF_NVIDIA_GEMV_CUH__
