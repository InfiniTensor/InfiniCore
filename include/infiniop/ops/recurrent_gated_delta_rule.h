#ifndef __INFINIOP_RECURRENT_GATED_DELTA_RULE_API_H__
#define __INFINIOP_RECURRENT_GATED_DELTA_RULE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRecurrentGatedDeltaRuleDescriptor_t;

__C __export infiniStatus_t infiniopCreateRecurrentGatedDeltaRuleDescriptor(
    infiniopHandle_t handle,
    infiniopRecurrentGatedDeltaRuleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    bool use_qk_l2norm);

__C __export infiniStatus_t infiniopGetRecurrentGatedDeltaRuleWorkspaceSize(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopRecurrentGatedDeltaRule(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc,
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

__C __export infiniStatus_t infiniopDestroyRecurrentGatedDeltaRuleDescriptor(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc);

#endif
