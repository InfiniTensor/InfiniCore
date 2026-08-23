#ifndef __INFINIOP_FP8_BLOCKWISE_DEQUANTIZE_API_H__
#define __INFINIOP_FP8_BLOCKWISE_DEQUANTIZE_API_H__

#include "../operator_descriptor.h"

/**
 * Dequantize a 2D FP8 E4M3FN blockwise-quantized weight tensor.
 *
 * The quantized input has shape [M, N] and dtype F8 (E4M3FN, stored as raw
 * bytes). Scales have shape [M / BM, N / BN] and dtype F32, where BM and BN
 * are the block sizes inferred from the shapes (typically 128 x 128). The
 * output has shape [M, N] and may be FP16, BF16, or FP32. All tensors must be
 * contiguous, M must be divisible by BM, and N must be divisible by BN.
 *
 * out[i, j] = fp8_e4m3_decode(q[i, j]) * scales[i / BM, j / BN]
 */
typedef struct InfiniopDescriptor *infiniopFp8BlockwiseDequantizeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFp8BlockwiseDequantizeDescriptor(
    infiniopHandle_t handle,
    infiniopFp8BlockwiseDequantizeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t scales_desc);

__INFINI_C __export infiniStatus_t infiniopGetFp8BlockwiseDequantizeWorkspaceSize(
    infiniopFp8BlockwiseDequantizeDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopFp8BlockwiseDequantize(
    infiniopFp8BlockwiseDequantizeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *scales,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFp8BlockwiseDequantizeDescriptor(
    infiniopFp8BlockwiseDequantizeDescriptor_t desc);

#endif
