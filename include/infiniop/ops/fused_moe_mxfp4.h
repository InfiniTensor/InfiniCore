#ifndef __INFINIOP_FUSED_MOE_MXFP4_API_H__
#define __INFINIOP_FUSED_MOE_MXFP4_API_H__

#include "../operator_descriptor.h"
#include "fused_moe.h"

/**
 * Fused routed A16 x MXFP4 MoE.
 *
 * input/output: [T, H] FP16/BF16/FP32
 * selected_experts: [T, topk] I32
 * routing_weights: [T, topk] F32
 * w13_packed/scales: [E, 2I, H/2] and [E, 2I, H/32] U8
 * w2_packed/scales: [E, H, I/2] and [E, H, I/32] U8
 */
typedef struct InfiniopDescriptor *infiniopFusedMoeMxfp4Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFusedMoeMxfp4Descriptor(
    infiniopHandle_t handle,
    infiniopFusedMoeMxfp4Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t selected_experts_desc,
    infiniopTensorDescriptor_t routing_weights_desc,
    infiniopTensorDescriptor_t w13_packed_desc,
    infiniopTensorDescriptor_t w13_scale_desc,
    infiniopTensorDescriptor_t w2_packed_desc,
    infiniopTensorDescriptor_t w2_scale_desc,
    infiniopFusedMoeActivation_t activation);

__INFINI_C __export infiniStatus_t infiniopGetFusedMoeMxfp4WorkspaceSize(
    infiniopFusedMoeMxfp4Descriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopFusedMoeMxfp4(
    infiniopFusedMoeMxfp4Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *selected_experts,
    const void *routing_weights,
    const void *w13_packed,
    const void *w13_scale,
    const void *w2_packed,
    const void *w2_scale,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFusedMoeMxfp4Descriptor(
    infiniopFusedMoeMxfp4Descriptor_t desc);

#endif
