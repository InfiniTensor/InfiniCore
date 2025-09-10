#ifndef __INFINIOP_SOFTPLUS_API_H__
#define __INFINIOP_SOFTPLUS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSoftplusDescriptor_t;

__C __export infiniStatus_t infiniopCreateSoftplusDescriptor(infiniopHandle_t handle,
                                                             infiniopSoftplusDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t y,
                                                             infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetSoftplusWorkspaceSize(infiniopSoftplusDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSoftplus(infiniopSoftplusDescriptor_t desc,
                                             void *workspace,
                                             size_t workspace_size,
                                             void *y,
                                             const void *x,
                                             void *stream);

__C __export infiniStatus_t infiniopDestroySoftplusDescriptor(infiniopSoftplusDescriptor_t desc);

#endif
