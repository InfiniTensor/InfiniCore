#ifndef __INFINIOP_MAXPOOL_BACKWARD_H__
#define __INFINIOP_MAXPOOL_BACKWARD_H__

#include "../operator_descriptor.h"

__C typedef struct InfiniopDescriptor *infiniopMaxPoolBackwardDescriptor_t;

__C infiniStatus_t infiniopCreateMaxPoolBackwardDescriptor(infiniopHandle_t handle,
                                                           infiniopMaxPoolBackwardDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t grad_input_desc,
                                                           infiniopTensorDescriptor_t grad_output_desc,
                                                           infiniopTensorDescriptor_t input_desc,
                                                           void *kernel_size,
                                                           void *strides,
                                                           void *pads,
                                                           bool ceil_mode);

__C infiniStatus_t infiniopGetMaxPoolBackwardWorkspaceSize(infiniopMaxPoolBackwardDescriptor_t desc,
                                                           size_t *size);

__C infiniStatus_t infiniopMaxPoolBackward(infiniopMaxPoolBackwardDescriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *grad_input,
                                           const void *grad_output,
                                           const void *input,
                                           void *stream);

__C infiniStatus_t infiniopDestroyMaxPoolBackwardDescriptor(infiniopMaxPoolBackwardDescriptor_t desc);

#endif // __INFINIOP_MAXPOOL_BACKWARD_H__
