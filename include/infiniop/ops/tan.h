#ifndef __INFINIOP_TAN_API_H__
#define __INFINIOP_TAN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTanDescriptor_t;

__C __export infiniStatus_t infiniopCreateTanDescriptor(infiniopHandle_t handle,
                                                        infiniopTanDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetTanWorkspaceSize(infiniopTanDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTan(infiniopTanDescriptor_t desc,
                                       void *workspace,
                                       size_t workspace_size,
                                       void *y,
                                       const void *x,
                                       void *stream);

__C __export infiniStatus_t infiniopDestroyTanDescriptor(infiniopTanDescriptor_t desc);

#endif
