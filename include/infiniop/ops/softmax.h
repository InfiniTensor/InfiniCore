#ifndef __INFINIOP_SOFTMAX_API_H__
#define __INFINIOP_SOFTMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSoftMaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateSoftMaxDescriptor(infiniopHandle_t handle,
                                                            infiniopSoftMaxDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y_desc,
                                                            infiniopTensorDescriptor_t x_desc,
                                                            int axis);

__C __export infiniStatus_t infiniopGetSoftMaxWorkspaceSize(infiniopSoftMaxDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSoftMax(infiniopSoftMaxDescriptor_t desc, void *workspace, size_t workspace_size, void *y, const void *x, void *stream);

__C __export infiniStatus_t infiniopDestroySoftMaxDescriptor(infiniopSoftMaxDescriptor_t desc);

#endif
