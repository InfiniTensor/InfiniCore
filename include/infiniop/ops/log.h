#ifndef __INFINIOP_LOG_API_H__
#define __INFINIOP_LOG_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogDescriptor_t;

__C __export infiniStatus_t infiniopCreateLogDescriptor(infiniopHandle_t handle,
                                                        infiniopLogDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetLogWorkspaceSize(infiniopLogDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLog(infiniopLogDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyLogDescriptor(infiniopLogDescriptor_t desc);

#endif
