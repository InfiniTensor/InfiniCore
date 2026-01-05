#ifndef __INFINIOP_MIN_API_H__
#define __INFINIOP_MIN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMinDescriptor_t;

__C __export infiniStatus_t infiniopCreateMinDescriptor(infiniopHandle_t handle,
                                                        infiniopMinDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetMinWorkspaceSize(infiniopMinDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMin(infiniopMinDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyMinDescriptor(infiniopMinDescriptor_t desc);

#endif
