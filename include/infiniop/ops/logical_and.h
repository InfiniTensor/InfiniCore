#ifndef __INFINIOP_LOGICAL_AND_API_H__
#define __INFINIOP_LOGICAL_AND_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogicalAndDescriptor_t;

__C __export infiniStatus_t infiniopCreateLogicalAndDescriptor(infiniopHandle_t handle,
                                                               infiniopLogicalAndDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t c,
                                                               infiniopTensorDescriptor_t a,
                                                               infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetLogicalAndWorkspaceSize(infiniopLogicalAndDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLogicalAnd(infiniopLogicalAndDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *c,
                                               const void *a,
                                               const void *b,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroyLogicalAndDescriptor(infiniopLogicalAndDescriptor_t desc);

#endif
