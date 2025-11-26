#ifndef __INFINIOP_BILINEAR_API_H__
#define __INFINIOP_BILINEAR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBilinearDescriptor_t;

__C __export infiniStatus_t infiniopCreateBilinearDescriptor(
    infiniopHandle_t handle,
    infiniopBilinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc); // bias 可以为 nullptr

__C __export infiniStatus_t infiniopGetBilinearWorkspaceSize(
    infiniopBilinearDescriptor_t desc, 
    size_t *size);

__C __export infiniStatus_t infiniopBilinear(
    infiniopBilinearDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void const *x1,
    void const *x2,
    void const *weight,
    void const *bias,
    void *stream);

__C __export infiniStatus_t infiniopDestroyBilinearDescriptor(
    infiniopBilinearDescriptor_t desc);

#endif