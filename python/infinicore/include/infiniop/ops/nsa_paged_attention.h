#ifndef __INFINIOP_NSA_PAGED_ATTENTION_API_H__
#define __INFINIOP_NSA_PAGED_ATTENTION_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopNsaPagedAttentionDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateNsaPagedAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopNsaPagedAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cmp_desc,
    infiniopTensorDescriptor_t v_cmp_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t gates_desc,
    float scale,
    int nsa_block_size,
    int window_size,
    int select_blocks);

__INFINI_C __export infiniStatus_t infiniopGetNsaPagedAttentionWorkspaceSize(
    infiniopNsaPagedAttentionDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopNsaPagedAttention(
    infiniopNsaPagedAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cmp,
    const void *v_cmp,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *gates,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyNsaPagedAttentionDescriptor(
    infiniopNsaPagedAttentionDescriptor_t desc);

#endif // __INFINIOP_NSA_PAGED_ATTENTION_API_H__
