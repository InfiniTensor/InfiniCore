// Prefill-path weight decoding for GGUF block-quantized weights, NVIDIA backend.
//
//   C[M, N] = A[M, K] @ W[N, K]^T,   M > kMaxDecodeM
//
// The register-resident GEMV kernel in linear_gguf_gemv.cuh gives up past
// kMaxDecodeM because it keeps one fp32 accumulator per input row in registers.
// A larger batch wants a real gemm instead, but `W` is still a packed GGUF blob,
// so launch_prefill below expands kPrefillTileN weight rows into a BF16 scratch at
// a time and hands that tile to cublas.  Only one tile is ever live -- a full
// dequantized copy of the weight would defeat the memory property that route B
// exists to keep.
//
// Tiling walks N rather than K: `C` is [M, N], so distinct tiles own distinct,
// non-overlapping output columns and every cublas call can use beta = 0.
// Accumulating along K instead would need an fp32 copy of `C` plus a conversion
// pass, for no benefit at these shapes.
//
// Block decoding is not duplicated here either: decode_block<TYPE> and
// BlockTraits come from linear_gguf_gemv.cuh, whose shared source ggml_blocks.h
// was verified bit-exactly against gguf-py on both host and device in stage 3.1.
//
// Status: correctness-first v1 (stage 3.3).  The scratch is written and then read
// back once per gemm, so prefill moves roughly 3x the traffic a fused
// decode-in-shared-memory kernel would; closing that gap is stage 6.1 (MMQ-style),
// not a precondition for correct results.
#ifndef __LINEAR_GGUF_NVIDIA_DEQUANT_CUH__
#define __LINEAR_GGUF_NVIDIA_DEQUANT_CUH__

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstddef>

#include "linear_gguf_gemv.cuh"

namespace op::linear_gguf::nvidia {

// Weight rows decoded per scratch tile.  64 keeps the scratch at
// 64 * K * 2 bytes (1.25 MiB at K = 10240) and hands cublas a [K, 64] operand
// whose leading dimension is still a multiple of 8 elements, as bf16 gemms want.
constexpr int kPrefillTileN = 64;
constexpr int kDequantThreads = 256;

// Bytes of BF16 scratch that Descriptor::calculate needs for one tile.  `k` is in
// elements, not packed bytes.
inline size_t prefill_scratch_bytes(int64_t k) {
    return static_cast<size_t>(kPrefillTileN) * static_cast<size_t>(k) * sizeof(__nv_bfloat16);
}

template <int32_t TYPE>
__global__ void dequant_tile_kernel(const uint8_t *__restrict__ w,
                                    __nv_bfloat16 *__restrict__ tile,
                                    int64_t n_start, int rows, int k, int64_t row_bytes) {
    constexpr int32_t kBytes = BlockTraits<TYPE>::kBytes;
    constexpr int32_t kElems = BlockTraits<TYPE>::kElems;

    const int blocks_per_row = k / kElems;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int r = static_cast<int>(idx / blocks_per_row);
    if (r >= rows) return;
    const int b = static_cast<int>(idx - static_cast<int64_t>(r) * blocks_per_row);

    float wk[kElems];
    decode_block<TYPE>(w + (n_start + static_cast<int64_t>(r)) * row_bytes
                           + static_cast<int64_t>(b) * kBytes,
                       wk);

    // kElems is even for every supported type and b * kElems is therefore a
    // 4-byte-aligned element offset, so pairs can be stored as one 32-bit write.
    uint32_t *out = reinterpret_cast<uint32_t *>(
        tile + static_cast<int64_t>(r) * k + static_cast<int64_t>(b) * kElems);
#pragma unroll
    for (int32_t j = 0; j < kElems; j += 2) {
        const __nv_bfloat162 pair = __floats2bfloat162_rn(wk[j], wk[j + 1]);
        out[j / 2] = *reinterpret_cast<const uint32_t *>(&pair);
    }
}

// Decode weight rows [n_start, n_start + rows) of `w` into `tile`, which is row
// major with leading dimension `k`.  Returns false without launching anything
// when `type` has no decoder or the geometry is not a whole number of blocks per
// row; the caller must not turn that into a dense fallback.
inline bool launch_dequant_tile(int32_t type, const uint8_t *w, __nv_bfloat16 *tile,
                                int64_t n_start, int rows, int k, int64_t row_bytes,
                                cudaStream_t stream) {
    const int32_t elems = ggml_blocks::block_elems(type);
    const int32_t bytes = ggml_blocks::block_bytes(type);
    if (rows <= 0 || k <= 0 || elems <= 0 || bytes <= 0) return false;
    if (n_start < 0) return false;
    if (k % elems != 0) return false;
    if (row_bytes < static_cast<int64_t>(k / elems) * bytes) return false;

    const int64_t total = static_cast<int64_t>(rows) * (k / elems);
    const unsigned grid = static_cast<unsigned>(
        (total + kDequantThreads - 1) / kDequantThreads);
    switch (type) {
    case ggml_blocks::GGML_TYPE_Q8_0:
        dequant_tile_kernel<ggml_blocks::GGML_TYPE_Q8_0><<<grid, kDequantThreads, 0, stream>>>(
            w, tile, n_start, rows, k, row_bytes);
        break;
    case ggml_blocks::GGML_TYPE_Q4_K:
        dequant_tile_kernel<ggml_blocks::GGML_TYPE_Q4_K><<<grid, kDequantThreads, 0, stream>>>(
            w, tile, n_start, rows, k, row_bytes);
        break;
    case ggml_blocks::GGML_TYPE_Q5_K:
        dequant_tile_kernel<ggml_blocks::GGML_TYPE_Q5_K><<<grid, kDequantThreads, 0, stream>>>(
            w, tile, n_start, rows, k, row_bytes);
        break;
    case ggml_blocks::GGML_TYPE_Q6_K:
        dequant_tile_kernel<ggml_blocks::GGML_TYPE_Q6_K><<<grid, kDequantThreads, 0, stream>>>(
            w, tile, n_start, rows, k, row_bytes);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

// c[m, n] = a[m, k] @ w[n, k]^T for m > kMaxDecodeM, one decoded weight tile at a
// time.  `scratch` / `scratch_bytes` come from the op's workspace tensor; the
// function refuses rather than silently allocating, so an undersized workspace is
// never mistaken for a wrong result.
//
// `blas` is the caller's cublas handle: the op borrows one from the device handle
// pool (devices/nvidia/nvidia_common.cu::useCublas) while
// scripts/gguf_routeb_gemv_probe.cu creates its own, so the numerical gate and the
// shipped path run this same composition.
//
// Tiling walks n, so each tile owns disjoint output columns and every gemm has
// beta = 0 -- no fp32 accumulator of `c` and no conversion pass.  The operand
// mapping (cublas is column-major, so row-major c[m, n] is [n, m]) is the same one
// ops/scaled_mm/nvidia/int8_gemm_nvidia.cu uses:
//   A = tile   stored [k, rows], op(A) = A^T -> [rows, k], lda = k
//   B = a      stored [k, m],    op(B) = B   -> [k, m],    ldb = k
//   C = c + n0 stored [n, m],                                    ldc = n
inline bool launch_prefill(cublasHandle_t blas, int32_t type,
                           const __nv_bfloat16 *a, const uint8_t *w, __nv_bfloat16 *c,
                           int m, int n, int k, int64_t row_bytes,
                           void *scratch, size_t scratch_bytes, cudaStream_t stream) {
    if (blas == nullptr || scratch == nullptr) return false;
    // The regular route uses this composition only for prefill.  Strict
    // compatibility experiments may deliberately route small-M decode here as
    // well, so validate only the actual matrix geometry.
    if (m <= 0 || n <= 0 || k <= 0) return false;
    if (scratch_bytes < prefill_scratch_bytes(k)) return false;

    auto *tile = reinterpret_cast<__nv_bfloat16 *>(scratch);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) return false;

    for (int n0 = 0; n0 < n; n0 += kPrefillTileN) {
        const int rows = std::min(kPrefillTileN, n - n0);
        if (!launch_dequant_tile(type, w, tile, n0, rows, k, row_bytes, stream)) return false;
        if (cublasGemmEx(blas,
                         CUBLAS_OP_T,   // A = tile^T : [rows, k]
                         CUBLAS_OP_N,   // B = a viewed column-major : [k, m]
                         rows, m, k,
                         &alpha,
                         tile, CUDA_R_16BF, k,
                         a, CUDA_R_16BF, k,
                         &beta,
                         c + n0, CUDA_R_16BF, n,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace op::linear_gguf::nvidia

#endif  // __LINEAR_GGUF_NVIDIA_DEQUANT_CUH__
