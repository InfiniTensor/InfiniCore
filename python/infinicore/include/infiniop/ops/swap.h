#ifndef __INFINIOP_SWAP_API_H__
#define __INFINIOP_SWAP_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSwapDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSwapDescriptor(infiniopHandle_t handle,
                                                                infiniopSwapDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t x,
                                                                infiniopTensorDescriptor_t y);

__INFINI_C __export infiniStatus_t infiniopGetSwapWorkspaceSize(infiniopSwapDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSwap(infiniopSwapDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *x,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySwapDescriptor(infiniopSwapDescriptor_t desc);

#endif // __INFINIOP_SWAP_API_H__
