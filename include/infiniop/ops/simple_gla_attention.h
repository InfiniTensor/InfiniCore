#ifndef __INFINIOP_SIMPLE_GLA_ATTENTION_API_H__
#define __INFINIOP_SIMPLE_GLA_ATTENTION_API_H__

#include "../operator_descriptor.h"

// Full-sequence Simple GLA attention forward (reference CPU + NVIDIA via prefill kernels).
// q, k, v: [B, T, H, D] (F32/F16/BF16), g_gamma: [H] (F32), out: [B, T, H, D] (same dtype as q)
typedef struct InfiniopDescriptor *infiniopSimpleGLAAttentionDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSimpleGLAAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopSimpleGLAAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_gamma_desc);

__INFINI_C __export infiniStatus_t infiniopGetSimpleGLAAttentionWorkspaceSize(
    infiniopSimpleGLAAttentionDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopSimpleGLAAttention(
    infiniopSimpleGLAAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void const *q,
    void const *k,
    void const *v,
    void const *g_gamma,
    float scale,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySimpleGLAAttentionDescriptor(
    infiniopSimpleGLAAttentionDescriptor_t desc);

#endif
