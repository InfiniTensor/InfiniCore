#ifndef __INFINIOP_CONV1D_API_H__
#define __INFINIOP_CONV1D_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopConv1dDescriptor_t;

typedef struct {
    infiniopConv1dDescriptor_t desc;
    void *y;
    const void *x_now;
    const void *w;
    void *conv_state;
    size_t B, C, K;
    infiniDtype_t dtype;
    void *stream;
} infiniopConv1dUpdateParams_t;

__C __export infiniStatus_t infiniopCreateConv1dDescriptor(infiniopHandle_t handle,
                                                           infiniopConv1dDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y_desc,
                                                           infiniopTensorDescriptor_t x_desc,
                                                           infiniopTensorDescriptor_t w_desc,
                                                           size_t kernel_size);

__C __export infiniStatus_t infiniopGetConv1dWorkspaceSize(infiniopConv1dDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopConv1dFn(infiniopConv1dDescriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *y,
                                              const void *x,
                                              const void *w,
                                              void *stream);

__C __export infiniStatus_t infiniopConv1dUpdate(infiniopConv1dUpdateParams_t *params);

__C __export infiniStatus_t infiniopDestroyConv1dDescriptor(infiniopConv1dDescriptor_t desc);

#endif