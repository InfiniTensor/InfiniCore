#ifndef __INFINIOP_BLOCK_FP8_LINEAR_API_H__
#define __INFINIOP_BLOCK_FP8_LINEAR_API_H__

#include "../operator_descriptor.h"

/**
 * Block-wise FP8 (E4M3) linear operation with dynamic activation quantization.
 *
 * output:     contiguous [M, N] BF16
 * input:      contiguous [M, K] BF16
 * weight:     contiguous [N, K] F8 (E4M3), used as logical B [K,N] column-major
 * weight_scale: contiguous [ceil(N/128), ceil(K/128)] F32 (dequantization scale)
 *
 * Internally: BF16 activation -> per-128-group dynamic quant to FP8 E4M3 ->
 *   SM120 CUTLASS blockwise scaled GEMM -> BF16 output.
 * Weight stays as 1 byte/element; no full-weight dequantization.
 */
typedef struct InfiniopDescriptor *infiniopBlockFP8LinearDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateBlockFP8LinearDescriptor(
    infiniopHandle_t handle,
    infiniopBlockFP8LinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc);

__INFINI_C __export infiniStatus_t infiniopGetBlockFP8LinearWorkspaceSize(
    infiniopBlockFP8LinearDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopBlockFP8Linear(
    infiniopBlockFP8LinearDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *weight_scale,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyBlockFP8LinearDescriptor(
    infiniopBlockFP8LinearDescriptor_t desc);

#endif
