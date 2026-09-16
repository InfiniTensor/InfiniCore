#ifndef __INFINIOP_LINEAR_W4A8_API_H__
#define __INFINIOP_LINEAR_W4A8_API_H__

#include "../operator_descriptor.h"

/**
 * Dynamically quantized A8 x W4 linear operation.
 *
 * input is contiguous [..., K] FP16/BF16/FP32. packed_weight is contiguous
 * [N, K / 2] I8, with the first logical value in the high nibble and signed
 * two's-complement INT4 values. weight_scale is contiguous [N, 1] F32,
 * optional bias is [N] with the input dtype, and output is [..., N].
 */
typedef struct InfiniopDescriptor *infiniopLinearW4A8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLinearW4A8Descriptor(
    infiniopHandle_t handle,
    infiniopLinearW4A8Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t packed_weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t bias_desc,
    float alpha);

__INFINI_C __export infiniStatus_t infiniopGetLinearW4A8WorkspaceSize(
    infiniopLinearW4A8Descriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopLinearW4A8(
    infiniopLinearW4A8Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *packed_weight,
    const void *weight_scale,
    const void *bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLinearW4A8Descriptor(
    infiniopLinearW4A8Descriptor_t desc);

#endif
