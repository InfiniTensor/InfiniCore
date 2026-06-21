#ifndef __INFINIOP_RWKV5_WKV_API_H__
#define __INFINIOP_RWKV5_WKV_API_H__

#include "../operator_descriptor.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

typedef struct InfiniopDescriptor *infiniopRwkv5WkvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRwkv5WkvDescriptor(
    infiniopHandle_t handle,
    infiniopRwkv5WkvDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t receptance_desc,
    infiniopTensorDescriptor_t key_desc,
    infiniopTensorDescriptor_t value_desc,
    infiniopTensorDescriptor_t time_decay_desc,
    infiniopTensorDescriptor_t time_faaaa_desc,
    infiniopTensorDescriptor_t state_desc);

__INFINI_C __export infiniStatus_t infiniopGetRwkv5WkvWorkspaceSize(
    infiniopRwkv5WkvDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopRwkv5Wkv(
    infiniopRwkv5WkvDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *receptance,
    const void *key,
    const void *value,
    const void *time_decay,
    const void *time_faaaa,
    void *state,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRwkv5WkvDescriptor(
    infiniopRwkv5WkvDescriptor_t desc);

#endif
