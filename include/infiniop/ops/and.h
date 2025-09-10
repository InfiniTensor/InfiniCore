#ifndef __INFINIOP_AND_API_H__
#define __INFINIOP_AND_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAndDescriptor_t;

__C __export infiniStatus_t infiniopCreateAndDescriptor(infiniopHandle_t handle,
                                                        infiniopAndDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetAndWorkspaceSize(infiniopAndDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAnd(infiniopAndDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyAndDescriptor(infiniopAndDescriptor_t desc);

#endif