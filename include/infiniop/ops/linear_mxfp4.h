#ifndef __INFINIOP_LINEAR_MXFP4_API_H__
#define __INFINIOP_LINEAR_MXFP4_API_H__

#include "../operator_descriptor.h"

/**
 * Fused A16 x MXFP4 linear operation.
 *
 * input is contiguous [..., K] FP16/BF16/FP32, packed_weight is contiguous
 * [N, K / 2] U8, weight_scale is contiguous [N, K / 32] U8 E8M0, optional
 * bias is [N] with the input dtype, and output is contiguous [..., N].
 */
typedef struct InfiniopDescriptor *infiniopLinearMxfp4Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLinearMxfp4Descriptor(
    infiniopHandle_t handle,
    infiniopLinearMxfp4Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t packed_weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t bias_desc,
    float alpha);

__INFINI_C __export infiniStatus_t infiniopGetLinearMxfp4WorkspaceSize(
    infiniopLinearMxfp4Descriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopLinearMxfp4(
    infiniopLinearMxfp4Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *packed_weight,
    const void *weight_scale,
    const void *bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLinearMxfp4Descriptor(
    infiniopLinearMxfp4Descriptor_t desc);

#endif
