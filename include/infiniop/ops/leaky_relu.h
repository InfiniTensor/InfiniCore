#ifndef __INFINIOP_LEAKY_RELU_API_H__
#define __INFINIOP_LEAKY_RELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLeakyReLUDescriptor_t;

__C __export infiniStatus_t infiniopCreateLeakyReLUDescriptor(infiniopHandle_t handle,
                                                              infiniopLeakyReLUDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t output,
                                                              infiniopTensorDescriptor_t input,
                                                              float negative_slope);

__C __export infiniStatus_t infiniopGetLeakyReLUWorkspaceSize(infiniopLeakyReLUDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLeakyReLU(infiniopLeakyReLUDescriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *output,
                                              const void *input,
                                              void *stream);

__C __export infiniStatus_t infiniopDestroyLeakyReLUDescriptor(infiniopLeakyReLUDescriptor_t desc);

#endif