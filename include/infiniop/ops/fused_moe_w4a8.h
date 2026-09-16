#ifndef __INFINIOP_FUSED_MOE_W4A8_API_H__
#define __INFINIOP_FUSED_MOE_W4A8_API_H__

#include "../operator_descriptor.h"
#include "fused_moe.h"

/**
 * Fused routed W4A8 MoE with dynamic per-token activation quantization.
 *
 * input/output: [T, H] FP16/BF16/FP32
 * selected_experts: [T, topk] I32
 * routing_weights: [T, topk] F32
 * w13_packed: [E, 2I, H/2] I8; w13_scale: [E, 2I, 1] F32
 * w2_packed: [E, H, I/2] I8; w2_scale: [E, H, 1] F32
 *
 * Each packed byte stores the earlier K value in its high nibble. Nibbles are
 * signed two's-complement INT4 values.
 * weights_are_aiter_shuffled selects the blocked AITER layout used by the
 * optimized Hygon backend. Other backends reject that layout.
 */
typedef struct InfiniopDescriptor *infiniopFusedMoeW4A8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFusedMoeW4A8Descriptor(
    infiniopHandle_t handle,
    infiniopFusedMoeW4A8Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t selected_experts_desc,
    infiniopTensorDescriptor_t routing_weights_desc,
    infiniopTensorDescriptor_t w13_packed_desc,
    infiniopTensorDescriptor_t w13_scale_desc,
    infiniopTensorDescriptor_t w2_packed_desc,
    infiniopTensorDescriptor_t w2_scale_desc,
    infiniopFusedMoeActivation_t activation,
    bool weights_are_aiter_shuffled);

__INFINI_C __export infiniStatus_t infiniopGetFusedMoeW4A8WorkspaceSize(
    infiniopFusedMoeW4A8Descriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopFusedMoeW4A8(
    infiniopFusedMoeW4A8Descriptor_t desc,
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

__INFINI_C __export infiniStatus_t infiniopDestroyFusedMoeW4A8Descriptor(
    infiniopFusedMoeW4A8Descriptor_t desc);

#endif
