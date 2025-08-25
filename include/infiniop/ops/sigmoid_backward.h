#ifndef __INFINIOP_SIGMOID_BACKWARD_API_H__
#define __INFINIOP_SIGMOID_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSigmoidBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateSigmoidBackwardDescriptor(infiniopHandle_t handle,
                                                                    infiniopSigmoidBackwardDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t grad_input,
                                                                    infiniopTensorDescriptor_t input,
                                                                    infiniopTensorDescriptor_t grad_output);

__C __export infiniStatus_t infiniopGetSigmoidBackwardWorkspaceSize(infiniopSigmoidBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSigmoidBackward(infiniopSigmoidBackwardDescriptor_t desc,
                                                   void *workspace,
                                                   size_t workspace_size,
                                                   void *grad_input,
                                                   const void *input,
                                                   const void *grad_output,
                                                   void *stream);

__C __export infiniStatus_t infiniopDestroySigmoidBackwardDescriptor(infiniopSigmoidBackwardDescriptor_t desc);

#endif