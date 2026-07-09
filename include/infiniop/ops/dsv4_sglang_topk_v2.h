#ifndef __INFINIOP_DSV4_SGLANG_TOPK_V2_H__
#define __INFINIOP_DSV4_SGLANG_TOPK_V2_H__
#include "../operator_descriptor.h"

#include <stdint.h>
typedef struct InfiniopDescriptor *infiniopDsv4SglangTopkV2Descriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateDsv4SglangTopkV2Descriptor(infiniopHandle_t handle, infiniopDsv4SglangTopkV2Descriptor_t *desc_ptr, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_table_desc, infiniopTensorDescriptor_t page_indices_desc, infiniopTensorDescriptor_t transform_workspace_desc, infiniopTensorDescriptor_t metadata_desc, int64_t page_size);
__INFINI_C __export infiniStatus_t infiniopGetDsv4SglangTopkV2WorkspaceSize(infiniopDsv4SglangTopkV2Descriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopDsv4SglangTopkV2(infiniopDsv4SglangTopkV2Descriptor_t desc, void *workspace, size_t workspace_size, const void *scores, const void *seq_lens, const void *page_table, void *page_indices, void *transform_workspace, void *metadata, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SglangTopkV2Descriptor(infiniopDsv4SglangTopkV2Descriptor_t desc);
#endif
