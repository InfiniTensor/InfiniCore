#ifndef __INFINIOP_FP8_INDEXER_LOGITS_API_H__
#define __INFINIOP_FP8_INDEXER_LOGITS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFp8IndexerLogitsDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFp8IndexerLogitsDescriptor(
    infiniopHandle_t handle,
    infiniopFp8IndexerLogitsDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t positions_desc,
    infiniopTensorDescriptor_t request_ids_desc);

__INFINI_C __export infiniStatus_t infiniopFp8IndexerLogits(
    infiniopFp8IndexerLogitsDescriptor_t desc,
    void *logits,
    const void *q_fp8,
    const void *kv_cache,
    const void *block_tables,
    const void *weights_fp32,
    const void *positions,
    const void *request_ids,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFp8IndexerLogitsDescriptor(
    infiniopFp8IndexerLogitsDescriptor_t desc);

#endif
