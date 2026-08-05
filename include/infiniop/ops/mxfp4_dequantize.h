#ifndef __INFINIOP_MXFP4_DEQUANTIZE_API_H__
#define __INFINIOP_MXFP4_DEQUANTIZE_API_H__

#include "../operator_descriptor.h"

/**
 * Dequantize an OCP MXFP4 tensor in the raw AMD Quark checkpoint layout.
 *
 * Logical values have shape [..., K]. The packed input is U8 [..., K / 2]
 * with the even logical value in the low nibble and the odd logical value in
 * the high nibble. Scales are U8 E8M0 values with shape [..., K / 32].
 * All tensors must be contiguous and K must be divisible by 32. Output may be
 * FP16, BF16, or FP32.
 */
typedef struct InfiniopDescriptor *infiniopMxfp4DequantizeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMxfp4DequantizeDescriptor(
    infiniopHandle_t handle,
    infiniopMxfp4DequantizeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t packed_desc,
    infiniopTensorDescriptor_t scales_desc);

__INFINI_C __export infiniStatus_t infiniopGetMxfp4DequantizeWorkspaceSize(
    infiniopMxfp4DequantizeDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMxfp4Dequantize(
    infiniopMxfp4DequantizeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *packed,
    const void *scales,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMxfp4DequantizeDescriptor(
    infiniopMxfp4DequantizeDescriptor_t desc);

#endif
