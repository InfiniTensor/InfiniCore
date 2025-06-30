#ifndef __INFINIOP_TIME_MIXING_API_H__
#define __INFINIOP_TIME_MIXING_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTimeMixingDescriptor_t;

__C __export infiniStatus_t infiniopCreateTimeMixingDescriptor(infiniopHandle_t handle,
                                                               infiniopTimeMixingDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t y_desc,
                                                               infiniopTensorDescriptor_t r_desc,
                                                               infiniopTensorDescriptor_t w_desc,
                                                               infiniopTensorDescriptor_t k_desc,
                                                               infiniopTensorDescriptor_t v_desc,
                                                               infiniopTensorDescriptor_t a_desc,
                                                               infiniopTensorDescriptor_t b_desc);

__C __export infiniStatus_t infiniopGetTimeMixingWorkspaceSize(infiniopTimeMixingDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTimeMixing(infiniopTimeMixingDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *y,
                                               void const *r,
                                               void const *w,
                                               void const *k,
                                               void const *v,
                                               void const *a,
                                               void const *b,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroyTimeMixingDescriptor(infiniopTimeMixingDescriptor_t desc);

#endif