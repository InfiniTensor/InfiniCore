#ifndef __INFINIOP_QUANT_API_H__
#define __INFINIOP_QUANT_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopQuantDescriptor_t;

__C __export infiniStatus_t infiniopCreateQuantDescriptor(infiniopHandle_t handle,
                                                          infiniopQuantDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t x_packed_desc,
                                                          infiniopTensorDescriptor_t x_scale_desc,
                                                          infiniopTensorDescriptor_t x_zero_desc,
                                                          infiniopTensorDescriptor_t x_desc);

__C __export infiniStatus_t infiniopGetQuantWorkspaceSize(infiniopQuantDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopQuant(infiniopQuantDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *x_packed,
                                          void *x_scale,
                                          void *x_zero,
                                          const void *x,
                                          void *stream);

__C __export infiniStatus_t infiniopDestroyQuantDescriptor(infiniopQuantDescriptor_t desc);

#endif
