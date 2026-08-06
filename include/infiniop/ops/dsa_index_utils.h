#ifndef __INFINIOP_DSA_INDEX_UTILS_API_H__
#define __INFINIOP_DSA_INDEX_UTILS_API_H__

#include "../operator_descriptor.h"

__INFINI_C __export infiniStatus_t infiniopMapDecodeRequestBlockIndices(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t request_ids_desc,
    infiniopTensorDescriptor_t block_table_desc,
    infiniopTensorDescriptor_t token_indices_desc,
    void *output,
    const void *request_ids,
    const void *block_table,
    const void *token_indices,
    int64_t block_size,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopTopkIndicesContextLens(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t topk_lens_desc,
    infiniopTensorDescriptor_t indices_desc,
    void *topk_lens,
    const void *indices,
    void *stream);

#endif
