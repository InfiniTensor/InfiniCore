#ifndef __INFINIOP_MAX_API_H__
#define __INFINIOP_MAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateMaxDescriptor(infiniopHandle_t handle,
                                                        infiniopMaxDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetMaxWorkspaceSize(infiniopMaxDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMax(infiniopMaxDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyMaxDescriptor(infiniopMaxDescriptor_t desc);

#endif
