#ifndef __INFINIOP_DEEPSEEK_MOE_W8A8I8_API_H__
#define __INFINIOP_DEEPSEEK_MOE_W8A8I8_API_H__

#include "../operator_descriptor.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

typedef struct InfiniopDescriptor *infiniopDeepseekMoeW8A8I8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekMoeW8A8I8Descriptor(
    infiniopHandle_t handle,
    infiniopDeepseekMoeW8A8I8Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t hidden_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    size_t intermediate_size,
    size_t num_experts);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekMoeW8A8I8WorkspaceSize(
    infiniopDeepseekMoeW8A8I8Descriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekMoeW8A8I8(
    infiniopDeepseekMoeW8A8I8Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    const void *const *gate_weight_scales,
    const void *const *up_weight_scales,
    const void *const *down_weight_scales,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDeepseekMoeW8A8I8WithDevicePtrs(
    infiniopDeepseekMoeW8A8I8Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *gate_weight_ptrs,
    const void *up_weight_ptrs,
    const void *down_weight_ptrs,
    const void *gate_weight_scale_ptrs,
    const void *up_weight_scale_ptrs,
    const void *down_weight_scale_ptrs,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekMoeW8A8I8Descriptor(
    infiniopDeepseekMoeW8A8I8Descriptor_t desc);

#endif
