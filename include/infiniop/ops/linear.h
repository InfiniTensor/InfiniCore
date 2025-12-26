#ifndef __INFINIOP_LINEAR_API_H__
#define __INFINIOP_LINEAR_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopLinearDescriptor_t;

__C __export infiniStatus_t infiniopCreateLinearDescriptor(infiniopHandle_t handle,
                                                           infiniopLinearDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t d_desc,
                                                           infiniopTensorDescriptor_t c_desc,
                                                           infiniopTensorDescriptor_t bias_desc,
                                                           infiniopTensorDescriptor_t x_desc,
                                                           infiniopTensorDescriptor_t x_scale_desc,
                                                           infiniopTensorDescriptor_t x_zero_desc,
                                                           infiniopTensorDescriptor_t weights_desc,
                                                           infiniopTensorDescriptor_t weights_scale_desc,
                                                           infiniopTensorDescriptor_t weights_zero_desc,
                                                           float alpha,
                                                           float beta);

__C __export infiniStatus_t infiniopGetLinearWorkspaceSize(infiniopLinearDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLinear(infiniopLinearDescriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *d,
                                           const void *c,
                                           const void *bias,
                                           const void *x,
                                           const void *x_scale,
                                           const void *x_zero,
                                           const void *weights,
                                           const void *weights_scale,
                                           const void *weights_zero,
                                           void *stream);

__C __export infiniStatus_t infiniopDestroyLinearDescriptor(infiniopLinearDescriptor_t desc);

#endif
