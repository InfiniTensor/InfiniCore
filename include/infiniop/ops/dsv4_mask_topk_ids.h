#ifndef __INFINIOP_DSV4_MASK_TOPK_IDS_API_H__
#define __INFINIOP_DSV4_MASK_TOPK_IDS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4MaskTopkIdsDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4MaskTopkIdsDescriptor(
    infiniopHandle_t handle,
    infiniopDsv4MaskTopkIdsDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_ids_desc,
    infiniopTensorDescriptor_t num_token_non_padded_desc);

__INFINI_C __export infiniStatus_t infiniopGetDsv4MaskTopkIdsWorkspaceSize(
    infiniopDsv4MaskTopkIdsDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4MaskTopkIds(
    infiniopDsv4MaskTopkIdsDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_ids,
    const void *num_token_non_padded,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4MaskTopkIdsDescriptor(
    infiniopDsv4MaskTopkIdsDescriptor_t desc);

#endif
