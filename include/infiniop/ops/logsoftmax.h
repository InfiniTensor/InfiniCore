#ifndef __INFINIOP_LOGSOFTMAX_API_H__
#define __INFINIOP_LOGSOFTMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogSoftmaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateLogSoftmaxDescriptor(infiniopHandle_t handle,
                                                               infiniopLogSoftmaxDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t y_desc,
                                                               infiniopTensorDescriptor_t x_desc);

__C __export infiniStatus_t infiniopGetLogSoftmaxWorkspaceSize(infiniopLogSoftmaxDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLogSoftmax(infiniopLogSoftmaxDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *y,
                                               const void *x,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroyLogSoftmaxDescriptor(infiniopLogSoftmaxDescriptor_t desc);

#endif
