#ifndef __INFINIOP_MOE_ALIGN_API_H__
#define __INFINIOP_MOE_ALIGN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoeAlignDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMoeAlignDescriptor(
    infiniopHandle_t handle,
    infiniopMoeAlignDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t expert_ids_desc,
    infiniopTensorDescriptor_t num_tokens_post_padded_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    size_t num_experts,
    size_t block_size);

__INFINI_C __export infiniStatus_t infiniopGetMoeAlignWorkspaceSize(
    infiniopMoeAlignDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMoeAlign(
    infiniopMoeAlignDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *sorted_token_ids,
    void *expert_ids,
    void *num_tokens_post_padded,
    const void *topk_ids,
    const void *expert_map,
    int pad_sorted_token_ids,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMoeAlignDescriptor(
    infiniopMoeAlignDescriptor_t desc);

#endif
