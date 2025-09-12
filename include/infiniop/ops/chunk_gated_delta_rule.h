#ifndef __INFINIOP_CHUNK_GATED_DELTA_RULE_API_H__
#define __INFINIOP_CHUNK_GATED_DELTA_RULE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopChunkGatedDeltaRuleDescriptor_t;

__C __export infiniStatus_t infiniopCreateChunkGatedDeltaRuleDescriptor(
    infiniopHandle_t handle,
    infiniopChunkGatedDeltaRuleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    bool use_qk_l2norm,
    size_t chunk_size);

__C __export infiniStatus_t infiniopGetChunkGatedDeltaRuleWorkspaceSize(
    infiniopChunkGatedDeltaRuleDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopChunkGatedDeltaRule(
    infiniopChunkGatedDeltaRuleDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *initial_state,
    void *stream);

__C __export infiniStatus_t infiniopDestroyChunkGatedDeltaRuleDescriptor(
    infiniopChunkGatedDeltaRuleDescriptor_t desc);

#endif
