#ifndef __INFINIOP_FP8_SPARSE_MLA_API_H__
#define __INFINIOP_FP8_SPARSE_MLA_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFp8SparseMlaDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFp8SparseMlaDescriptor(
    infiniopHandle_t handle,
    infiniopFp8SparseMlaDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t topk_lens_desc,
    float scale);

__INFINI_C __export infiniStatus_t infiniopGetFp8SparseMlaWorkspaceSize(
    infiniopFp8SparseMlaDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopFp8SparseMla(
    infiniopFp8SparseMlaDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *query,
    const void *kv_cache,
    const void *indices,
    const void *topk_lens,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFp8SparseMlaDescriptor(
    infiniopFp8SparseMlaDescriptor_t desc);

#endif
