#ifndef __INFINIOP_DEEPSEEK_MOE_API_H__
#define __INFINIOP_DEEPSEEK_MOE_API_H__

#include "../operator_descriptor.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

typedef struct InfiniopDescriptor *infiniopDeepseekMoeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekMoeDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekMoeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t hidden_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    size_t intermediate_size,
    size_t num_experts);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekMoeWorkspaceSize(
    infiniopDeepseekMoeDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekMoe(
    infiniopDeepseekMoeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDeepseekMoeWithDevicePtrs(
    infiniopDeepseekMoeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *gate_weight_ptrs,
    const void *up_weight_ptrs,
    const void *down_weight_ptrs,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekMoeDescriptor(
    infiniopDeepseekMoeDescriptor_t desc);

#endif
