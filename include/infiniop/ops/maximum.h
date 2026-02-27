#ifndef __INFINIOP_MAXIMUM_API_H__
#define __INFINIOP_MAXIMUM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMaximumDescriptor_t;

__C __export infiniStatus_t infiniopCreateMaximumDescriptor(infiniopHandle_t handle,
                                                            infiniopMaximumDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t c,
                                                            infiniopTensorDescriptor_t a,
                                                            infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetMaximumWorkspaceSize(infiniopMaximumDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMaximum(infiniopMaximumDescriptor_t desc,
                                            void *workspace,
                                            size_t workspace_size,
                                            void *c,
                                            const void *a,
                                            const void *b,
                                            void *stream);

__C __export infiniStatus_t infiniopDestroyMaximumDescriptor(infiniopMaximumDescriptor_t desc);

#endif
