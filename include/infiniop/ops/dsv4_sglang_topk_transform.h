#ifndef __INFINIOP_DSV4_SGLANG_TOPK_TRANSFORM_H__
#define __INFINIOP_DSV4_SGLANG_TOPK_TRANSFORM_H__

#include "../operator_descriptor.h"
#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopDsv4SglangTopkTransformDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4SglangTopkTransformDescriptor(infiniopHandle_t handle,
                                                                                   infiniopDsv4SglangTopkTransformDescriptor_t *desc_ptr,
                                                                                   infiniopTensorDescriptor_t scores_desc,
                                                                                   infiniopTensorDescriptor_t seq_lens_desc,
                                                                                   infiniopTensorDescriptor_t page_table_desc,
                                                                                   infiniopTensorDescriptor_t page_indices_desc,
                                                                                   infiniopTensorDescriptor_t raw_indices_desc,
                                                                                   int64_t page_size);

__INFINI_C __export infiniStatus_t infiniopGetDsv4SglangTopkTransformWorkspaceSize(infiniopDsv4SglangTopkTransformDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4SglangTopkTransform(infiniopDsv4SglangTopkTransformDescriptor_t desc,
                                                                   void *workspace,
                                                                   size_t workspace_size,
                                                                   const void *scores,
                                                                   const void *seq_lens,
                                                                   const void *page_table,
                                                                   void *page_indices,
                                                                   void *raw_indices,
                                                                   void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SglangTopkTransformDescriptor(infiniopDsv4SglangTopkTransformDescriptor_t desc);

#endif
