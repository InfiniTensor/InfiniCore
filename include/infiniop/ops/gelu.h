#ifndef __INFINIOP_GELU_API_H__
#define __INFINIOP_GELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGeluDescriptor_t;

__C __export infiniStatus_t infiniopCreateGeluDescriptor(infiniopHandle_t handle,
                                                         infiniopGeluDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t output,
                                                         infiniopTensorDescriptor_t input);

__C __export infiniStatus_t infiniopGetGeluWorkspaceSize(infiniopGeluDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGelu(infiniopGeluDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *output,
                                         const void *input,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyGeluDescriptor(infiniopGeluDescriptor_t desc);

#endif