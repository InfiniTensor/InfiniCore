#ifndef __INFINIOP_FLASH_ATTENTION_BACKWARD_H__
#define __INFINIOP_FLASH_ATTENTION_BACKWARD_H__

#include "../operator_descriptor.h"
#include "flash_attention.h"

typedef struct InfiniopDescriptor *infiniopFlashAttentionBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateFlashAttentionBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_q_desc,
    infiniopTensorDescriptor_t grad_k_desc,
    infiniopTensorDescriptor_t grad_v_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t grad_out_desc,
    infiniopTensorDescriptor_t mask_desc,
    infiniopAttentionMaskType_t mask_type);

__C __export infiniStatus_t infiniopGetFlashAttentionBackwardWorkspaceSize(
    infiniopFlashAttentionBackwardDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopFlashAttentionBackward(
    infiniopFlashAttentionBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_q,
    void *grad_k,
    void *grad_v,
    const void *q,
    const void *k,
    const void *v,
    const void *grad_out,
    const void *mask,
    void *stream);

__C __export infiniStatus_t infiniopDestroyFlashAttentionBackwardDescriptor(
    infiniopFlashAttentionBackwardDescriptor_t desc);

#endif
