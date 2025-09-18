#ifndef __INFINIOP_FLASH_ATTENTION_API_H__
#define __INFINIOP_FLASH_ATTENTION_API_H__

#include "../operator_descriptor.h"

typedef enum {
    INFINIOP_ATTENTION_MASK_TYPE_NONE = 0,
    INFINIOP_ATTENTION_MASK_TYPE_FULL = 1,
    INFINIOP_ATTENTION_MASK_TYPE_CAUSAL = 2,
} infiniopAttentionMaskType_t;

typedef struct InfiniopDescriptor *infiniopFlashAttentionDescriptor_t;

__C __export infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t l_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t mask_desc,
    infiniopAttentionMaskType_t mask_type);

__C __export infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopFlashAttention(
    infiniopFlashAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void *l,
    const void *q,
    const void *k,
    const void *v,
    const void *mask,
    void *stream);

__C __export infiniStatus_t infiniopDestroyFlashAttentionDescriptor(infiniopFlashAttentionDescriptor_t desc);

#endif
