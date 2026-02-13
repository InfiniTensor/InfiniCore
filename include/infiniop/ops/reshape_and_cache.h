#ifndef __INFINIOP_RESHAPE_AND_CACHE_API_H__
#define __INFINIOP_RESHAPE_AND_CACHE_API_H__

#include "../operator_descriptor.h"
#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopReshapeAndCacheDescriptor_t;

__C __export infiniStatus_t infiniopCreateReshapeAndCacheDescriptor(
    infiniopHandle_t handle,
    infiniopReshapeAndCacheDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t key_desc,
    infiniopTensorDescriptor_t value_desc,
    infiniopTensorDescriptor_t key_cache_desc,
    infiniopTensorDescriptor_t value_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    const char *kv_cache_dtype);

__C __export infiniStatus_t infiniopGetReshapeAndCacheWorkspaceSize(
    infiniopReshapeAndCacheDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopReshapeAndCache(
    infiniopReshapeAndCacheDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *key,
    void *value,
    void *key_cache,
    void *value_cache,
    const void *slot_mapping,
    const char *kv_cache_dtype,
    void *k_scale,
    void *v_scale,
    void *stream);

__C __export infiniStatus_t infiniopDestroyReshapeAndCacheDescriptor(
    infiniopReshapeAndCacheDescriptor_t desc);

#endif // __INFINIOP_RESHAPE_AND_CACHE_API_H__
