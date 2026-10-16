#ifndef __INFINIOP_MOE_ARGSORT_BINCOUNT_API_H__
#define __INFINIOP_MOE_ARGSORT_BINCOUNT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoeArgsortBincountDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMoeArgsortBincountDescriptor(
    infiniopHandle_t handle,
    infiniopMoeArgsortBincountDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t tokens_per_experts_desc,
    infiniopTensorDescriptor_t sorted_indices_desc,
    infiniopTensorDescriptor_t inv_pos_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    size_t num_experts);

__INFINI_C __export infiniStatus_t infiniopGetMoeArgsortBincountWorkspaceSize(
    infiniopMoeArgsortBincountDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMoeArgsortBincount(
    infiniopMoeArgsortBincountDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *tokens_per_experts,
    void *sorted_indices,
    void *inv_pos,
    const void *topk_ids,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMoeArgsortBincountDescriptor(
    infiniopMoeArgsortBincountDescriptor_t desc);

#endif
