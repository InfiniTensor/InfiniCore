#ifndef __INFINIOP_EQUAL_API_H__
#define __INFINIOP_EQUAL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogicalEqualDescriptor_t;

__C __export infiniStatus_t infiniopCreateLogicalEqualDescriptor(infiniopHandle_t handel,
                                                                 infiniopLogicalEqualDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t c,
                                                                 infiniopTensorDescriptor_t a,
                                                                 infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetLogicalEqualWorkspaceSize(infiniopLogicalEqualDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLogicalEqual(infiniopLogicalEqualDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *c,
                                                 const void *a,
                                                 const void *b,
                                                 void *stream);

__C __export infiniStatus_t infiniopDestroyLogicalEqualDescriptor(infiniopLogicalEqualDescriptor_t desc);
#endif
