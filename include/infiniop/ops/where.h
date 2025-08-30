#ifndef __INFINIOP_WHERE_API_H__
#define __INFINIOP_WHERE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopWhereDescriptor_t;

__C __export infiniStatus_t infiniopCreateWhereDescriptor(infiniopHandle_t handle,
                                                          infiniopWhereDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c,
                                                          infiniopTensorDescriptor_t condition,
                                                          infiniopTensorDescriptor_t a,
                                                          infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetWhereWorkspaceSize(infiniopWhereDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopWhere(infiniopWhereDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *c,
                                          const void *condition,
                                          const void *a,
                                          const void *b,
                                          void *stream);

__C __export infiniStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc);

#endif