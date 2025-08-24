#ifndef __INFINIOP_ROPE_V2_API_H__
#define __INFINIOP_ROPE_V2_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRoPEv2Descriptor_t;

__C __export infiniStatus_t infiniopCreateRoPEv2Descriptor(
    infiniopHandle_t handle,
    infiniopRoPEv2Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table);

__C __export infiniStatus_t infiniopGetRoPEv2WorkspaceSize(infiniopRoPEv2Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopRoPEv2(
    infiniopRoPEv2Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void *stream);

__C __export infiniStatus_t infiniopDestroyRoPEv2Descriptor(infiniopRoPEv2Descriptor_t desc);

#endif
