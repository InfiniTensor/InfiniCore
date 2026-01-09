#ifndef __INFINIOP_NEG_API_H__
#define __INFINIOP_NEG_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopNegDescriptor_t;

__C __export infiniStatus_t infiniopCreateNegDescriptor(infiniopHandle_t handle,
                                                        infiniopNegDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetNegWorkspaceSize(infiniopNegDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopNeg(infiniopNegDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyNegDescriptor(infiniopNegDescriptor_t desc);

#endif
