#ifndef __INFINIOP_3DMROPE_API_H__
#define __INFINIOP_3DMROPE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMRoPE3DDescriptor_t;

__C __export infiniStatus_t infiniopCreateMRoPE3DDescriptor(
    infiniopHandle_t handle,
    infiniopMRoPE3DDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table,
    infiniopTensorDescriptor_t rope_section);

__C __export infiniStatus_t infiniopGetMRoPE3DWorkspaceSize(infiniopMRoPE3DDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMRoPE3D(
    infiniopMRoPE3DDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void const *rope_section,
    void *stream);

__C __export infiniStatus_t infiniopDestroyMRoPE3DDescriptor(infiniopMRoPE3DDescriptor_t desc);

#endif
