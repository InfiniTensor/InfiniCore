#ifndef __INFINIOP_KIMI_DELTA_ATTENTION_API_H__
#define __INFINIOP_KIMI_DELTA_ATTENTION_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopKimiDeltaAttentionDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateKimiDeltaAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopKimiDeltaAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,                   // [B,T,H,D] or varlen [1,total_tokens,H,D]
    infiniopTensorDescriptor_t initial_state_desc,         // [B,H,D,D] or indexed pool [pool_size,H,D,D]
    infiniopTensorDescriptor_t final_state_desc,           // null when final_state_indices_desc is provided
    infiniopTensorDescriptor_t q_desc,                     // [B,T,H,D] or varlen [1,total_tokens,H,D]
    infiniopTensorDescriptor_t k_desc,                     // same shape as q
    infiniopTensorDescriptor_t v_desc,                     // same shape as q
    infiniopTensorDescriptor_t g_desc,                     // raw KDA gate [B,T,H,D] or varlen [1,total_tokens,H,D]
    infiniopTensorDescriptor_t beta_desc,                  // raw beta logits [B,T,H] or varlen [1,total_tokens,H]
    infiniopTensorDescriptor_t A_log_desc,                 // [H], fp32
    infiniopTensorDescriptor_t dt_bias_desc,               // [H,D], fp32
    infiniopTensorDescriptor_t cu_seqlens_desc,            // nullable; [B + 1], int32/int64
    infiniopTensorDescriptor_t initial_state_indices_desc, // nullable; [B], int32/int64
    infiniopTensorDescriptor_t final_state_indices_desc,   // nullable; [B], int32/int64; writes final state in-place to initial_state
    float scale,
    float lower_bound,
    bool use_qk_l2norm);

__INFINI_C __export infiniStatus_t infiniopGetKimiDeltaAttentionWorkspaceSize(
    infiniopKimiDeltaAttentionDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopKimiDeltaAttention(
    infiniopKimiDeltaAttentionDescriptor_t desc,
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
    const void *A_log,
    const void *dt_bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyKimiDeltaAttentionDescriptor(
    infiniopKimiDeltaAttentionDescriptor_t desc);

#endif
