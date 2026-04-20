#ifndef __INFINIOP_SIMPLE_GLA_PREFILL_API_H__
#define __INFINIOP_SIMPLE_GLA_PREFILL_API_H__

#include "../operator_descriptor.h"

// Chunked/fused Simple GLA prefill forward.
// q, k, v: [B, T, H, D] (F16/BF16), g_gamma: [H] (F32), out: [B, T, H, D] (same dtype as q)
typedef struct InfiniopDescriptor *infiniopSimpleGLAPrefillDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSimpleGLAPrefillDescriptor(
    infiniopHandle_t handle,
    infiniopSimpleGLAPrefillDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_gamma_desc);

__INFINI_C __export infiniStatus_t infiniopGetSimpleGLAPrefillWorkspaceSize(
    infiniopSimpleGLAPrefillDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopSimpleGLAPrefill(
    infiniopSimpleGLAPrefillDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void const *q,
    void const *k,
    void const *v,
    void const *g_gamma,
    float scale,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySimpleGLAPrefillDescriptor(
    infiniopSimpleGLAPrefillDescriptor_t desc);

#endif

