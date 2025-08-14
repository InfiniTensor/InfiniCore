#ifndef __INFINIOP_RELU_BACKWARD_API_H__
#define __INFINIOP_RELU_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopReluBackWardDescriptor_t;

__C __export infiniStatus_t infiniopCreateReluBackWardDescriptor(infiniopHandle_t handle,
                                                                 infiniopReluBackWardDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t grad_input,
                                                                 infiniopTensorDescriptor_t input,
                                                                 infiniopTensorDescriptor_t grad_output);

__C __export infiniStatus_t infiniopGetReluBackWardWorkspaceSize(infiniopReluBackWardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopReluBackWard(infiniopReluBackWardDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *grad_input,
                                                 const void *input,
                                                 const void *grad_output,
                                                 void *stream);

__C __export infiniStatus_t infiniopDestroyReluBackWardDescriptor(infiniopReluBackWardDescriptor_t desc);

#endif
