#ifndef __INFINIOP_OR_API_H__
#define __INFINIOP_OR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopOrDescriptor_t;

__C __export infiniStatus_t infiniopCreateOrDescriptor(infiniopHandle_t handle,
                                                        infiniopOrDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetOrWorkspaceSize(infiniopOrDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopOr(infiniopOrDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyOrDescriptor(infiniopOrDescriptor_t desc);

#endif