#ifndef __INFINIOP_CONV_BACKWARD_API_H__
#define __INFINIOP_CONV_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopConvBackwardDescriptor_t;

__C infiniStatus_t infiniopCreateConvBackwardDescriptor(infiniopHandle_t handle,
                                                        infiniopConvBackwardDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t grad_output_desc,
                                                        infiniopTensorDescriptor_t input_desc,
                                                        infiniopTensorDescriptor_t weight_desc,
                                                        infiniopTensorDescriptor_t bias_desc,
                                                        void *pads,
                                                        void *strides,
                                                        void *dilations,
                                                        size_t n);

__C infiniStatus_t infiniopGetConvBackwardWorkspaceSize(infiniopConvBackwardDescriptor_t desc, size_t *size);

__C infiniStatus_t infiniopConvBackward(infiniopConvBackwardDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *grad_input,
                                        void *grad_weight,
                                        void *grad_bias,
                                        const void *grad_output,
                                        const void *input,
                                        const void *weight,
                                        void *stream);

__C infiniStatus_t infiniopDestroyConvBackwardDescriptor(infiniopConvBackwardDescriptor_t desc);

#endif
