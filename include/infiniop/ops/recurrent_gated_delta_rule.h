#ifndef __INFINIOP_RECURRENT_GATED_DELTA_RULE_API_H__
#define __INFINIOP_RECURRENT_GATED_DELTA_RULE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRecurrentGatedDeltaRuleDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRecurrentGatedDeltaRuleDescriptor(
    infiniopHandle_t handle,
    infiniopRecurrentGatedDeltaRuleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,                   // [B, T, Hv, Dv], T must be 1; last dim contiguous
    infiniopTensorDescriptor_t initial_state_desc,         // legacy: [B, Hv, Dv, Dk]; indexed pool: [pool_size, Hv, Dv, Dk]
    infiniopTensorDescriptor_t final_state_desc,           // legacy/indexed out-of-place final state; null when final_state_indices_desc is provided
    infiniopTensorDescriptor_t q_desc,                     // [B, T, Hk, Dk], T must be 1; last dim contiguous
    infiniopTensorDescriptor_t k_desc,                     // [B, T, Hk, Dk], same shape as q; last dim contiguous
    infiniopTensorDescriptor_t v_desc,                     // [B, T, Hv, Dv], Hv must be a multiple of Hk; last dim contiguous
    infiniopTensorDescriptor_t g_desc,                     // [B, T, Hv]; may have a different fp dtype from q/k/v/out/state
    infiniopTensorDescriptor_t beta_desc,                  // [B, T, Hv]; same dtype as g
    infiniopTensorDescriptor_t initial_state_indices_desc, // nullable; [B], int32/int64; enables indexed pool mode
    infiniopTensorDescriptor_t final_state_indices_desc,   // nullable; [B], int32/int64; writes final state in-place to initial_state pool
    bool use_qk_l2norm);

__INFINI_C __export infiniStatus_t infiniopGetRecurrentGatedDeltaRuleWorkspaceSize(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopRecurrentGatedDeltaRule(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRecurrentGatedDeltaRuleDescriptor(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc);

#endif
