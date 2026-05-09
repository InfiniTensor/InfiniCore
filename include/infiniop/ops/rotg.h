#ifndef __INFINIOP_ROTG_API_H__
#define __INFINIOP_ROTG_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRotgDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRotgDescriptor(infiniopHandle_t handle,
                                                                infiniopRotgDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t x,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t c,
                                                                infiniopTensorDescriptor_t s);

__INFINI_C __export infiniStatus_t infiniopGetRotgWorkspaceSize(infiniopRotgDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopRotg(infiniopRotgDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *x,
                                                void *y,
                                                void *c,
                                                void *s,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRotgDescriptor(infiniopRotgDescriptor_t desc);

#endif // __INFINIOP_ROTG_API_H__
