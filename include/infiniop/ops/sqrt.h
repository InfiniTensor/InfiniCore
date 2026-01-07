#ifndef __INFINIOP_SQRT_API_H__
#define __INFINIOP_SQRT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSqrtDescriptor_t;

__C __export infiniStatus_t infiniopCreateSqrtDescriptor(infiniopHandle_t handle,
                                                         infiniopSqrtDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y,
                                                         infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetSqrtWorkspaceSize(infiniopSqrtDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSqrt(infiniopSqrtDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroySqrtDescriptor(infiniopSqrtDescriptor_t desc);

#endif
