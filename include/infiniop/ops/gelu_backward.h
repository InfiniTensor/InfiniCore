#ifndef __INFINIOP_GELU_BACKWARD_API_H__
#define __INFINIOP_GELU_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGeLUBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateGeLUBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopGeLUBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t grad_output_desc
);

__C __export infiniStatus_t infiniopGetGeLUBackwardWorkspaceSize(infiniopGeLUBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGeLUBackward(
    infiniopGeLUBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * grad_input,
    const void * input,
    const void * grad_output,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyGeLUBackwardDescriptor(infiniopGeLUBackwardDescriptor_t desc);

#endif
