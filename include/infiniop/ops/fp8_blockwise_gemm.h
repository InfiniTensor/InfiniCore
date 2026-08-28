#ifndef __INFINIOP_FP8_BLOCKWISE_GEMM_API_H__
#define __INFINIOP_FP8_BLOCKWISE_GEMM_API_H__

#include "../operator_descriptor.h"

/**
 * Fused GEMM for FP8 E4M3FN blockwise-quantized weights (decode-oriented).
 *
 * Computes out = a @ dequantize(q, scales)^T without materializing the
 * dequantized weight:
 *
 *   out[m, n] = sum_k a[m, k] * fp8_e4m3_decode(q[n, k]) * scales[n / BN, k / BK]
 *
 * - out: [M, N], dtype F16/BF16/F32, contiguous
 * - a:   [M, K], same dtype as out, contiguous
 * - q:   [N, K], dtype F8 (E4M3FN raw bytes), contiguous
 * - scales: [N / BN, K / BK], dtype F32, contiguous (typically BN = BK = 128)
 *
 * N must be divisible by the scale row count and K by the scale col count.
 * The operator is optimized for small M (decode); large M still works but is
 * not the target use case.
 */
typedef struct InfiniopDescriptor *infiniopFp8BlockwiseGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFp8BlockwiseGemmDescriptor(
    infiniopHandle_t handle,
    infiniopFp8BlockwiseGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t scales_desc);

__INFINI_C __export infiniStatus_t infiniopGetFp8BlockwiseGemmWorkspaceSize(
    infiniopFp8BlockwiseGemmDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopFp8BlockwiseGemm(
    infiniopFp8BlockwiseGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *q,
    const void *scales,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFp8BlockwiseGemmDescriptor(
    infiniopFp8BlockwiseGemmDescriptor_t desc);

#endif
