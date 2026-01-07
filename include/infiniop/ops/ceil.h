#ifndef __INFINIOP_CEIL_API_H__
#define __INFINIOP_CEIL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCeilDescriptor_t;

__C __export infiniStatus_t infiniopCreateCeilDescriptor(infiniopHandle_t handle,
                                                        infiniopCeilDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetCeilWorkspaceSize(infiniopCeilDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopCeil(infiniopCeilDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyCeilDescriptor(infiniopCeilDescriptor_t desc);

#endif
