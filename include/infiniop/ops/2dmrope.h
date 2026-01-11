#ifndef __INFINIOP_2DMROPE_API_H__
#define __INFINIOP_2DMROPE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMRoPE2DDescriptor_t;

__C __export infiniStatus_t infiniopCreateMRoPE2DDescriptor(
    infiniopHandle_t handle,
    infiniopMRoPE2DDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table);

__C __export infiniStatus_t infiniopGetMRoPE2DWorkspaceSize(infiniopMRoPE2DDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMRoPE2D(
    infiniopMRoPE2DDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void *stream);

__C __export infiniStatus_t infiniopDestroyMRoPE2DDescriptor(infiniopMRoPE2DDescriptor_t desc);

#endif
