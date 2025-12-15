#ifndef __INFINIOP_VDOT_API_H__
#define __INFINIOP_VDOT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopVdotDescriptor_t;

// out = vdot(a, b)
__C __export infiniStatus_t infiniopCreateVdotDescriptor(
    infiniopHandle_t handle,
    infiniopVdotDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc);

__C __export infiniStatus_t infiniopGetVdotWorkspaceSize(
    infiniopVdotDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopVdot(
    infiniopVdotDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    void *stream);

__C __export infiniStatus_t infiniopDestroyVdotDescriptor(
    infiniopVdotDescriptor_t desc);

#endif // __INFINIOP_VDOT_API_H__


