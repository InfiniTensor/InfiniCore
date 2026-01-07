#ifndef __INFINIOP_ROUND_API_H__
#define __INFINIOP_ROUND_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRoundDescriptor_t;

__C __export infiniStatus_t infiniopCreateRoundDescriptor(infiniopHandle_t handle,
                                                           infiniopRoundDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetRoundWorkspaceSize(infiniopRoundDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopRound(infiniopRoundDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *y,
                                          const void *x,
                                          void *stream);

__C __export infiniStatus_t infiniopDestroyRoundDescriptor(infiniopRoundDescriptor_t desc);

#endif
