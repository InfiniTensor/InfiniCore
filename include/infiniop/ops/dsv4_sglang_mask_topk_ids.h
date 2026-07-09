#ifndef __INFINIOP_DSV4_SGLANG_MASK_TOPK_IDS_H__
#define __INFINIOP_DSV4_SGLANG_MASK_TOPK_IDS_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4SglangMaskTopkIdsDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4SglangMaskTopkIdsDescriptor(infiniopHandle_t handle,
                                                                                 infiniopDsv4SglangMaskTopkIdsDescriptor_t *desc_ptr,
                                                                                 infiniopTensorDescriptor_t topk_ids_desc,
                                                                                 infiniopTensorDescriptor_t num_token_non_padded_desc);

__INFINI_C __export infiniStatus_t infiniopGetDsv4SglangMaskTopkIdsWorkspaceSize(infiniopDsv4SglangMaskTopkIdsDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4SglangMaskTopkIds(infiniopDsv4SglangMaskTopkIdsDescriptor_t desc,
                                                                 void *workspace,
                                                                 size_t workspace_size,
                                                                 void *topk_ids,
                                                                 const void *num_token_non_padded,
                                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SglangMaskTopkIdsDescriptor(infiniopDsv4SglangMaskTopkIdsDescriptor_t desc);

#endif
