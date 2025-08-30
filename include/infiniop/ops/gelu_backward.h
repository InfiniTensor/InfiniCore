#ifndef __INFINIOP_GELU_BACKWARD_API_H__
#define __INFINIOP_GELU_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGeluBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateGeluBackwardDescriptor(infiniopHandle_t handle,
                                                                infiniopGeluBackwardDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t grad_input,
                                                                infiniopTensorDescriptor_t input,
                                                                infiniopTensorDescriptor_t grad_output);

__C __export infiniStatus_t infiniopGetGeluBackwardWorkspaceSize(infiniopGeluBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGeluBackward(infiniopGeluBackwardDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *grad_input,
                                                const void *input,
                                                const void *grad_output,
                                                void *stream);

__C __export infiniStatus_t infiniopDestroyGeluBackwardDescriptor(infiniopGeluBackwardDescriptor_t desc);

#endif