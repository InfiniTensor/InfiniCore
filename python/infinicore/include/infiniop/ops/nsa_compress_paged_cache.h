#ifndef __INFINIOP_NSA_COMPRESS_PAGED_CACHE_API_H__
#define __INFINIOP_NSA_COMPRESS_PAGED_CACHE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopNsaCompressPagedCacheDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateNsaCompressPagedCacheDescriptor(
    infiniopHandle_t handle,
    infiniopNsaCompressPagedCacheDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t k_cmp_desc,
    infiniopTensorDescriptor_t v_cmp_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    int nsa_block_size,
    int update_last_only);

__INFINI_C __export infiniStatus_t infiniopGetNsaCompressPagedCacheWorkspaceSize(
    infiniopNsaCompressPagedCacheDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopNsaCompressPagedCache(
    infiniopNsaCompressPagedCacheDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *k_cmp,
    void *v_cmp,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyNsaCompressPagedCacheDescriptor(
    infiniopNsaCompressPagedCacheDescriptor_t desc);

#endif // __INFINIOP_NSA_COMPRESS_PAGED_CACHE_API_H__
