#ifndef __INFINIOP_CHUNK_GATED_DELTA_RULE_API_H__
#define __INFINIOP_CHUNK_GATED_DELTA_RULE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopChunkGatedDeltaRuleDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateChunkGatedDeltaRuleDescriptor(
    infiniopHandle_t handle,
    infiniopChunkGatedDeltaRuleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,                   // padded: [B, T, Hv, Dv]; varlen: [1, total_tokens, Hv, Dv]
    infiniopTensorDescriptor_t initial_state_desc,         // legacy: [B, Hv, Dv, Dk]; indexed pool: [pool_size, Hv, Dv, Dk]
    infiniopTensorDescriptor_t final_state_desc,           // null when final_state_indices_desc is provided
    infiniopTensorDescriptor_t q_desc,                     // padded: [B, T, Hk, Dk]; varlen: [1, total_tokens, Hk, Dk]
    infiniopTensorDescriptor_t k_desc,                     // same shape as q
    infiniopTensorDescriptor_t v_desc,                     // padded: [B, T, Hv, Dv]; varlen: [1, total_tokens, Hv, Dv]
    infiniopTensorDescriptor_t g_desc,                     // padded: [B, T, Hv]; varlen: [1, total_tokens, Hv]
    infiniopTensorDescriptor_t beta_desc,                  // same shape/dtype as g
    infiniopTensorDescriptor_t cu_seqlens_desc,            // nullable; [B + 1], int32/int64
    infiniopTensorDescriptor_t initial_state_indices_desc, // nullable; [B], int32/int64; enables indexed state-pool reads
    infiniopTensorDescriptor_t final_state_indices_desc,   // nullable; [B], int32/int64; writes final state in-place to initial_state pool
    bool use_qk_l2norm,
    size_t chunk_size);

__INFINI_C __export infiniStatus_t infiniopGetChunkGatedDeltaRuleWorkspaceSize(
    infiniopChunkGatedDeltaRuleDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopChunkGatedDeltaRule(
    infiniopChunkGatedDeltaRuleDescriptor_t desc,
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
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyChunkGatedDeltaRuleDescriptor(
    infiniopChunkGatedDeltaRuleDescriptor_t desc);

#endif
