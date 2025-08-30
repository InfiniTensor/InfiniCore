#ifndef __INFINIOP_REDUCE_MEAN_API_H__
#define __INFINIOP_REDUCE_MEAN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopReduceMeanDescriptor_t;

__C __export infiniStatus_t infiniopCreateReduceMeanDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMeanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t dim);

__C __export infiniStatus_t infiniopGetReduceMeanWorkspaceSize(infiniopReduceMeanDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopReduceMean(
    infiniopReduceMeanDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

__C __export infiniStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceMeanDescriptor_t desc);

#endif
