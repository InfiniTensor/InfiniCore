#ifndef __INFINIOP_LEAKYRELU_API_H__
#define __INFINIOP_LEAKYRELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLeakyreluDescriptor_t;

__C __export infiniStatus_t infiniopCreateLeakyreluDescriptor(infiniopHandle_t handle,
                                                              infiniopLeakyreluDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t output,
                                                              infiniopTensorDescriptor_t input,
                                                              float negative_slope);

__C __export infiniStatus_t infiniopGetLeakyreluWorkspaceSize(infiniopLeakyreluDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLeakyrelu(infiniopLeakyreluDescriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *output,
                                              const void *input,
                                              void *stream);

__C __export infiniStatus_t infiniopDestroyLeakyreluDescriptor(infiniopLeakyreluDescriptor_t desc);

#endif
