#ifndef __INFINIOP_DEEPSEEK_V4_INDEXER_API_H__
#define __INFINIOP_DEEPSEEK_V4_INDEXER_API_H__

#include "../operator_descriptor.h"

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct InfiniopDescriptor *infiniopDeepseekV4IndexerDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4IndexerDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4IndexerDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t weights_desc,
    infiniopTensorDescriptor_t compressed_desc,
    infiniopTensorDescriptor_t positions_desc,
    size_t query_start,
    size_t compress_ratio);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4IndexerWorkspaceSize(
    infiniopDeepseekV4IndexerDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4Indexer(
    infiniopDeepseekV4IndexerDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *indices,
    const void *q,
    const void *weights,
    const void *compressed,
    const void *positions,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4IndexerDescriptor(
    infiniopDeepseekV4IndexerDescriptor_t desc);

#if defined(__cplusplus)
}
#endif

#endif
