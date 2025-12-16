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

// where(cond) -> indices tuple
typedef struct InfiniopDescriptor *infiniopWhereIndicesDescriptor_t;

__C __export infiniStatus_t infiniopCreateWhereIndicesDescriptor(
    infiniopHandle_t handle,
    infiniopWhereIndicesDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t cond_desc);

__C __export infiniStatus_t infiniopGetWhereIndicesWorkspaceSize(
    infiniopWhereIndicesDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopWhereIndices(
    infiniopWhereIndicesDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void **outputs,  // NDIM 个输出张量的指针数组
    const void *cond,
    void *stream,
    size_t *num_true);  // 输出：True 元素的数量

__C __export infiniStatus_t infiniopDestroyWhereIndicesDescriptor(
    infiniopWhereIndicesDescriptor_t desc);

#endif


