#ifndef __INFINIOP_ROT_API_H__
#define __INFINIOP_ROT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRotDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRotDescriptor(infiniopHandle_t handle,
                                                               infiniopRotDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t x,
                                                               infiniopTensorDescriptor_t y,
                                                               infiniopTensorDescriptor_t c,
                                                               infiniopTensorDescriptor_t s);

__INFINI_C __export infiniStatus_t infiniopGetRotWorkspaceSize(infiniopRotDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopRot(infiniopRotDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *x,
                                               void *y,
                                               const void *c,
                                               const void *s,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRotDescriptor(infiniopRotDescriptor_t desc);

#endif // __INFINIOP_ROT_API_H__
