#ifndef __INFINIOP_INNER_API_H__
#define __INFINIOP_INNER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopInnerDescriptor_t;

__C __export infiniStatus_t infiniopCreateInnerDescriptor(
    infiniopHandle_t handle,
    infiniopInnerDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t other_desc,
    infiniopTensorDescriptor_t out_desc);

__C __export infiniStatus_t infiniopGetInnerWorkspaceSize(infiniopInnerDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopInner(
    infiniopInnerDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *input,
    const void *other,
    void *out,
    void *stream);

__C __export infiniStatus_t infiniopDestroyInnerDescriptor(infiniopInnerDescriptor_t desc);

#endif