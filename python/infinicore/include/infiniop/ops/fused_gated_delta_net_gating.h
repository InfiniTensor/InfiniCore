#ifndef __INFINIOP_FUSED_GATED_DELTA_NET_GATING_API_H__
#define __INFINIOP_FUSED_GATED_DELTA_NET_GATING_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFusedGatedDeltaNetGatingDescriptor_t;

__INFINI_C __export infiniStatus_t
infiniopCreateFusedGatedDeltaNetGatingDescriptor(
    infiniopHandle_t handle,
    infiniopFusedGatedDeltaNetGatingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_output_desc,
    infiniopTensorDescriptor_t A_log_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t dt_bias_desc,
    float beta,
    float threshold);

__INFINI_C __export infiniStatus_t
infiniopGetFusedGatedDeltaNetGatingWorkspaceSize(
    infiniopFusedGatedDeltaNetGatingDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t
infiniopFusedGatedDeltaNetGating(
    infiniopFusedGatedDeltaNetGatingDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *g,
    void *beta_output,
    const void *A_log,
    const void *a,
    const void *b,
    const void *dt_bias,
    void *stream);

__INFINI_C __export infiniStatus_t
infiniopDestroyFusedGatedDeltaNetGatingDescriptor(
    infiniopFusedGatedDeltaNetGatingDescriptor_t desc);

#endif
