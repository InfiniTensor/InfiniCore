#ifndef __INFINIOP_DSV4_FLASH_MLA_DECODE_H__
#define __INFINIOP_DSV4_FLASH_MLA_DECODE_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4FlashMlaDecodeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4FlashMlaDecodeDescriptor(infiniopHandle_t handle,
                                                                              infiniopDsv4FlashMlaDecodeDescriptor_t *desc_ptr,
                                                                              infiniopTensorDescriptor_t out_desc,
                                                                              infiniopTensorDescriptor_t lse_desc,
                                                                              infiniopTensorDescriptor_t q_nope_desc,
                                                                              infiniopTensorDescriptor_t q_pe_desc,
                                                                              infiniopTensorDescriptor_t k_cache_desc,
                                                                              infiniopTensorDescriptor_t block_table_desc,
                                                                              infiniopTensorDescriptor_t cache_seqlens_desc,
                                                                              infiniopTensorDescriptor_t tile_scheduler_metadata_desc,
                                                                              infiniopTensorDescriptor_t num_splits_desc,
                                                                              int head_dim_v,
                                                                              float softmax_scale,
                                                                              bool causal);

__INFINI_C __export infiniStatus_t infiniopGetDsv4FlashMlaDecodeWorkspaceSize(infiniopDsv4FlashMlaDecodeDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4FlashMlaDecode(infiniopDsv4FlashMlaDecodeDescriptor_t desc,
                                                              void *workspace,
                                                              size_t workspace_size,
                                                              void *out,
                                                              void *lse,
                                                              const void *q_nope,
                                                              const void *q_pe,
                                                              const void *k_cache,
                                                              const void *block_table,
                                                              const void *cache_seqlens,
                                                              const void *tile_scheduler_metadata,
                                                              const void *num_splits,
                                                              void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4FlashMlaDecodeDescriptor(infiniopDsv4FlashMlaDecodeDescriptor_t desc);

#endif
