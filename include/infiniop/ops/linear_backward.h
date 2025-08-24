#ifndef __INFINIOP_LINEAR_BACKWARD_API_H__
#define __INFINIOP_LINEAR_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLinearBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateLinearBackwardDescriptor(infiniopHandle_t handle,
                                                                    infiniopLinearBackwardDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t grad_y,
                                                                    infiniopTensorDescriptor_t x,
                                                                    infiniopTensorDescriptor_t w,
                                                                    infiniopTensorDescriptor_t grad_x,
                                                                    infiniopTensorDescriptor_t grad_w,
                                                                    infiniopTensorDescriptor_t grad_b);

__C __export infiniStatus_t infiniopGetLinearBackwardWorkspaceSize(infiniopLinearBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLinearBackward(infiniopLinearBackwardDescriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *grad_x,
                                                    void *grad_w,
                                                    void *grad_b,
                                                    const void *grad_y,
                                                    const void *x,
                                                    const void *w,
                                                    void *stream);

__C __export infiniStatus_t infiniopDestroyLinearBackwardDescriptor(infiniopLinearBackwardDescriptor_t desc);

#endif