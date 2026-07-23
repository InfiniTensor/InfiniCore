#ifndef __INFINIOP_CAUSAL_CONV1D_API_H__
#define __INFINIOP_CAUSAL_CONV1D_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCausalConv1dDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateCausalConv1dDescriptor(
    infiniopHandle_t handle,
    infiniopCausalConv1dDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,                   // padded: [B, T, C]; varlen: [1, total_tokens, C]
    infiniopTensorDescriptor_t conv_state_desc,            // no-index: [B/num_requests, C, state_len]; pool: [pool_size, C, state_len]
    infiniopTensorDescriptor_t final_conv_state_desc,      // nullable when final_state_indices_desc is provided
    infiniopTensorDescriptor_t qkv_desc,                   // padded: [B, T, C]; varlen: [1, total_tokens, C]
    infiniopTensorDescriptor_t weight_desc,                // [C, 1, K], depthwise; current backend supports K == 4
    infiniopTensorDescriptor_t bias_desc,                  // nullable; [C]
    infiniopTensorDescriptor_t cu_seqlens_desc,            // nullable; [num_requests + 1], int32/int64
    infiniopTensorDescriptor_t initial_state_indices_desc, // nullable; [num_requests], int32/int64
    infiniopTensorDescriptor_t final_state_indices_desc);  // nullable; [num_requests], int32/int64; writes final state in-place

__INFINI_C __export infiniStatus_t infiniopGetCausalConv1dWorkspaceSize(
    infiniopCausalConv1dDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopCausalConv1d(
    infiniopCausalConv1dDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void *conv_state,
    void *final_conv_state,
    const void *qkv,
    const void *weight,
    const void *bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyCausalConv1dDescriptor(
    infiniopCausalConv1dDescriptor_t desc);

#endif // __INFINIOP_CAUSAL_CONV1D_API_H__
