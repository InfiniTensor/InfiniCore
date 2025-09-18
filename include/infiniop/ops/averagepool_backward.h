#ifndef __INFINIOP_AVERAGEPOOL_BACKWARD_H__
#define __INFINIOP_AVERAGEPOOL_BACKWARD_H__

#include "../operator_descriptor.h"

__C typedef struct InfiniopDescriptor *infiniopAvgPoolBackwardDescriptor_t;

__C infiniStatus_t infiniopCreateAvgPoolBackwardDescriptor(infiniopHandle_t handle,
                                                           infiniopAvgPoolBackwardDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t grad_input_desc,
                                                           infiniopTensorDescriptor_t grad_output_desc,
                                                           infiniopTensorDescriptor_t input_desc,
                                                           void *kernel_size,
                                                           void *strides,
                                                           void *pads,
                                                           bool ceil_mode);

__C infiniStatus_t infiniopGetAvgPoolBackwardWorkspaceSize(infiniopAvgPoolBackwardDescriptor_t desc,
                                                           size_t *size);

__C infiniStatus_t infiniopAvgPoolBackward(infiniopAvgPoolBackwardDescriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *grad_input,
                                           const void *grad_output,
                                           const void *input,
                                           void *stream);

__C infiniStatus_t infiniopDestroyAvgPoolBackwardDescriptor(infiniopAvgPoolBackwardDescriptor_t desc);

#endif // __INFINIOP_AVERAGEPOOL_BACKWARD_H__
