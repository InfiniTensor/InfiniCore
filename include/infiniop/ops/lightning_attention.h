#ifndef __INFINIOP_LIGHTNING_ATTENTION_API_H__
#define __INFINIOP_LIGHTNING_ATTENTION_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLightningAttentionDescriptor_t;

// Lightning attention (MiniMax-01 style, ALiBi-style per-head decay) with an
// indexed recurrent-state pool.
//
// Recurrence (per head h, per token t):
//   S  = ratio[h] * S + k_t^T v_t          (ratio[h] = exp(-slope[h]))
//   o_t = q_t @ S
// i.e. the state is updated *before* the output is read, so each output token
// attends to itself with weight 1 (no decay within the same position).
//
// Tensor layouts:
//   out                  [B, T, H, D]      (last dim contiguous)
//   initial_state (pool) [pool_size, H, D, D]
//   q/k/v                [B, T, H, D]      (last dim contiguous)
//   slope                [H]               (fp32)
//   initial/final_state_indices [B]        (int32 or int64)
//
// Indexed-pool mode only: for each request b the op reads the state row
// `initial_state[initial_state_indices[b]]` and writes the final state in place
// to `initial_state[final_state_indices[b]]`.
__INFINI_C __export infiniStatus_t infiniopCreateLightningAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopLightningAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slope_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc);

__INFINI_C __export infiniStatus_t infiniopGetLightningAttentionWorkspaceSize(
    infiniopLightningAttentionDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopLightningAttention(
    infiniopLightningAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void *initial_state,
    const void *q,
    const void *k,
    const void *v,
    const void *slope,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLightningAttentionDescriptor(
    infiniopLightningAttentionDescriptor_t desc);

#endif
