#ifndef __INFINIOP_DSV4_FLASH_MLA_METADATA_H__
#define __INFINIOP_DSV4_FLASH_MLA_METADATA_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4FlashMlaMetadataDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4FlashMlaMetadataDescriptor(infiniopHandle_t handle,
                                                                                infiniopDsv4FlashMlaMetadataDescriptor_t *desc_ptr,
                                                                                infiniopTensorDescriptor_t cache_seqlens_desc,
                                                                                infiniopTensorDescriptor_t tile_scheduler_metadata_desc,
                                                                                infiniopTensorDescriptor_t num_splits_desc,
                                                                                int num_heads_per_head_k,
                                                                                int num_heads_k);

__INFINI_C __export infiniStatus_t infiniopGetDsv4FlashMlaMetadataWorkspaceSize(infiniopDsv4FlashMlaMetadataDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4FlashMlaMetadata(infiniopDsv4FlashMlaMetadataDescriptor_t desc,
                                                                void *workspace,
                                                                size_t workspace_size,
                                                                const void *cache_seqlens,
                                                                void *tile_scheduler_metadata,
                                                                void *num_splits,
                                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4FlashMlaMetadataDescriptor(infiniopDsv4FlashMlaMetadataDescriptor_t desc);

#endif
