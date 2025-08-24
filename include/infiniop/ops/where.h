#ifndef __INFINIOP_WHERE_API_H__
#define __INFINIOP_WHERE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopWhereDescriptor_t;

__C __export infiniStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t condition_desc);

__C __export infiniStatus_t infiniopGetWhereWorkspaceSize(
    infiniopWhereDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopWhere(
    infiniopWhereDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    const void *condition,
    void *stream);

#endif // __INFINIOP_WHERE_API_H__
