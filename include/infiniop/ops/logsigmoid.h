#ifndef __INFINIOP_LOGSIGMOID_API_H__
#define __INFINIOP_LOGSIGMOID_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogSigmoidDescriptor_t;

__C __export infiniStatus_t infiniopCreateLogSigmoidDescriptor(infiniopHandle_t handle,
                                                                 infiniopLogSigmoidDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetLogSigmoidWorkspaceSize(infiniopLogSigmoidDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLogSigmoid(infiniopLogSigmoidDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x,
                                                void *stream);

__C __export infiniStatus_t infiniopDestroyLogSigmoidDescriptor(infiniopLogSigmoidDescriptor_t desc);

#endif

