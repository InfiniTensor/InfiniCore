#ifndef __INFINIOP_GELU_API_H__
#define __INFINIOP_GELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGeLUDescriptor_t;

__C __export infiniStatus_t infiniopCreateGeLUDescriptor(
    infiniopHandle_t handle,
    infiniopGeLUDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc
);

__C __export infiniStatus_t infiniopGetGeLUWorkspaceSize(infiniopGeLUDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGeLU(
    infiniopGeLUDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * output,
    const void * input,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyGeLUDescriptor(infiniopGeLUDescriptor_t desc);

#endif
