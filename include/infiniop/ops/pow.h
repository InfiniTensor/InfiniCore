#ifndef __INFINIOP_POW_API_H__
#define __INFINIOP_POW_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopPowDescriptor_t;

__C __export infiniStatus_t infiniopCreatePowDescriptor(infiniopHandle_t handle,
                                                        infiniopPowDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetPowWorkspaceSize(infiniopPowDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopPow(infiniopPowDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyPowDescriptor(infiniopPowDescriptor_t desc);

#endif
