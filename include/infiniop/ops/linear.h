#ifndef __INFINIOP_LINEAR_API_H__
#define __INFINIOP_LINEAR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLinearDescriptor_t;

__C __export infiniStatus_t infiniopCreateLinearDescriptor(
    infiniopHandle_t handle, infiniopLinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t d_desc, infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc, infiniopTensorDescriptor_t c_desc);

__C __export infiniStatus_t
infiniopGetLinearWorkspaceSize(infiniopLinearDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLinear(
    infiniopLinearDescriptor_t desc, float alpha, const void *a,
    const void *a_scale, const void *b, const void *b_scale, float beta,
    const void *c, const void *c_scale, const void *bias, void *d,
    const void *d_scale, bool is_blockwise, bool is_a_1d_scaled,
    bool is_b_1d_scaled, void *workspace, size_t workspace_size, void *stream);

__C __export infiniStatus_t
infiniopDestroyLinearDescriptor(infiniopLinearDescriptor_t desc);

#endif