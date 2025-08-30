#ifndef __INFINIOP_RELU_BACKWARD_API_H__
#define __INFINIOP_RELU_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopReLUBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateReLUBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopReLUBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t grad_output_desc
);

__C __export infiniStatus_t infiniopGetReLUBackwardWorkspaceSize(infiniopReLUBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopReLUBackward(
    infiniopReLUBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * grad_input,
    const void * input,
    const void * grad_output,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyReLUBackwardDescriptor(infiniopReLUBackwardDescriptor_t desc);

#endif
