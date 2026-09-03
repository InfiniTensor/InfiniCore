#ifndef __INFINIOP_LINEAR_GGUF_API_H__
#define __INFINIOP_LINEAR_GGUF_API_H__

#include "../operator_descriptor.h"
#include <cstdint>

/**
 * Linear over GGML block-quantized weights:
 *     out[M, N] = a[M, K] @ dequant(weight)[N, K]^T
 *
 * output:  contiguous [M, N] BF16
 * input:   contiguous [M, K] BF16
 * weight:  contiguous [N, row_bytes] U8 -- the GGML block rows of one tensor,
 *          packed back to back verbatim, so
 *          row_bytes == (K / block_elems(ggml_type)) * block_bytes(ggml_type)
 * ggml_type: enum ggml_type id of the weight blocks.  Supported: 8 (Q8_0),
 *          12 (Q4_K), 13 (Q5_K), 14 (Q6_K).  Any other id is rejected here.
 *
 * The weight stays in its quantized form: blocks are decoded inside the kernel
 * and accumulated in fp32, so a model loaded this way never materializes a
 * dense copy of its weights.
 *
 * Current backend implements the decode (GEMV) path only, i.e. M must not
 * exceed kMaxDecodeM from the NVIDIA kernel header.  A larger M returns
 * INFINI_STATUS_NOT_IMPLEMENTED rather than silently dequantizing the weight.
 */
typedef struct InfiniopDescriptor *infiniopLinearGgufDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLinearGgufDescriptor(infiniopHandle_t handle,
                                                                      infiniopLinearGgufDescriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t out_desc,
                                                                      infiniopTensorDescriptor_t a_desc,
                                                                      infiniopTensorDescriptor_t w_desc,
                                                                      int64_t ggml_type);

__INFINI_C __export infiniStatus_t infiniopGetLinearGgufWorkspaceSize(infiniopLinearGgufDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLinearGguf(infiniopLinearGgufDescriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *out,
                                                     const void *a,
                                                     const void *w,
                                                     void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLinearGgufDescriptor(infiniopLinearGgufDescriptor_t desc);

#endif  // __INFINIOP_LINEAR_GGUF_API_H__
