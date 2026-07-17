#ifndef __INFINIOP_FUSED_MOE_API_H__
#define __INFINIOP_FUSED_MOE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFusedMoeDescriptor_t;

typedef enum {
    INFINIOP_FUSED_MOE_ACT_SILU = 0,
    INFINIOP_FUSED_MOE_ACT_SWIGLU = 1,
} infiniopFusedMoeActivation_t;

__INFINI_C __export infiniStatus_t infiniopCreateFusedMoeDescriptor(
    infiniopHandle_t handle,
    infiniopFusedMoeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t token_selected_experts_desc,
    infiniopTensorDescriptor_t token_final_scales_desc,
    infiniopTensorDescriptor_t w1_desc,
    infiniopTensorDescriptor_t w2_desc,
    infiniopTensorDescriptor_t b1_desc,
    infiniopTensorDescriptor_t b2_desc,
    infiniopFusedMoeActivation_t activation);

__INFINI_C __export infiniStatus_t infiniopGetFusedMoeWorkspaceSize(
    infiniopFusedMoeDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopFusedMoe(
    infiniopFusedMoeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *input,
    const void *token_selected_experts,
    const void *token_final_scales,
    const void *w1,
    const void *w2,
    const void *b1,
    const void *b2,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFusedMoeDescriptor(
    infiniopFusedMoeDescriptor_t desc);

#endif // __INFINIOP_FUSED_MOE_API_H__
