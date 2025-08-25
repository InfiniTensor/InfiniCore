#ifndef __INFINIOP_GELU_BACKWARD_API_H__
#define __INFINIOP_GELU_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGeluBackWardDescriptor_t;

__C __export infiniStatus_t infiniopCreateGeluBackWardDescriptor(infiniopHandle_t handle,
                                                                 infiniopGeluBackWardDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t grad_input,
                                                                 infiniopTensorDescriptor_t input,
                                                                 infiniopTensorDescriptor_t grad_output);

__C __export infiniStatus_t infiniopGeluBackWard(infiniopGeluBackWardDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *grad_input,
                                                 const void *input,
                                                 const void *grad_output,
                                                 void *stream);

__C __export infiniStatus_t infiniopDestroyGeluBackWardDescriptor(infiniopGeluBackWardDescriptor_t desc);

#endif
