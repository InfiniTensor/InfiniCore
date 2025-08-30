#ifndef __INFINIOP_HARDSWISH_API_H__
#define __INFINIOP_HARDSWISH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopHardswishDescriptor_t;

__C __export infiniStatus_t infiniopCreateHardswishDescriptor(infiniopHandle_t handle,
                                                              infiniopHardswishDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t output,
                                                              infiniopTensorDescriptor_t input);

__C __export infiniStatus_t infiniopGetHardswishWorkspaceSize(infiniopHardswishDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopHardswish(infiniopHardswishDescriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *output,
                                              const void *input,
                                              void *stream);

__C __export infiniStatus_t infiniopDestroyHardswishDescriptor(infiniopHardswishDescriptor_t desc);

#endif
