#ifndef __INFINIOP_WHERE_API_H__
#define __INFINIOP_WHERE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopWhereDescriptor_t;

// y = where(cond, x, y)
__C __export infiniStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t cond_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc);

__C __export infiniStatus_t infiniopGetWhereWorkspaceSize(
    infiniopWhereDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopWhere(
    infiniopWhereDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *cond,
    const void *x,
    const void *y,
    void *stream);

__C __export infiniStatus_t infiniopDestroyWhereDescriptor(
    infiniopWhereDescriptor_t desc);

#endif


