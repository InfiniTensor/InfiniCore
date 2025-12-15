#ifndef __INFINIOP_LOGICAL_OR_API_H__
#define __INFINIOP_LOGICAL_OR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogicalOrDescriptor_t;

__C __export infiniStatus_t infiniopCreateLogicalOrDescriptor(infiniopHandle_t handle,
                                                                 infiniopLogicalOrDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t c_desc,
                                                                 infiniopTensorDescriptor_t a_desc,
                                                                 infiniopTensorDescriptor_t b_desc);

__C __export infiniStatus_t infiniopGetLogicalOrWorkspaceSize(infiniopLogicalOrDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLogicalOr(infiniopLogicalOrDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *c,
                                               const void *a,
                                               const void *b,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroyLogicalOrDescriptor(infiniopLogicalOrDescriptor_t desc);

#endif

