#ifndef __INFINIOP_RELU_BACKWARD_API_H__
#define __INFINIOP_RELU_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopReluBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateReluBackwardDescriptor(infiniopHandle_t handle,
                                                                 infiniopReluBackwardDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t grad_input,
                                                                 infiniopTensorDescriptor_t input,
                                                                 infiniopTensorDescriptor_t grad_output);

__C __export infiniStatus_t infiniopGetReluBackwardWorkspaceSize(infiniopReluBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopReluBackward(infiniopReluBackwardDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *grad_input,
                                                 const void *input,
                                                 const void *grad_output,
                                                 void *stream);

__C __export infiniStatus_t infiniopDestroyReluBackwardDescriptor(infiniopReluBackwardDescriptor_t desc);

#endif