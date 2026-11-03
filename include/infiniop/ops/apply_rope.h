#ifndef __INFINIOP_APPLY_ROPE_API_H__
#define __INFINIOP_APPLY_ROPE_API_H__
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopApplyRoPEDescriptor_t;
// Ascend NeoX rotation, in place. All tensors must be contiguous, with the
// same F16/BF16/F32 dtype. Q/K are [T,H,D], cos/sin [T,1,D], D=64 or 128.
// Q and K buffers must not overlap. The supplied cos/sin repeat both halves.
__INFINI_C __export infiniStatus_t infiniopCreateApplyRoPEDescriptor(
    infiniopHandle_t handle, infiniopApplyRoPEDescriptor_t *desc,
    infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin);
__INFINI_C __export infiniStatus_t infiniopGetApplyRoPEWorkspaceSize(
    infiniopApplyRoPEDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopApplyRoPE(
    infiniopApplyRoPEDescriptor_t desc, void *workspace, size_t workspace_size,
    void *q, void *k, const void *cos, const void *sin, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyApplyRoPEDescriptor(
    infiniopApplyRoPEDescriptor_t desc);
#endif
