#ifndef __INFINIOP_FUSED_FFN_API_H__
#define __INFINIOP_FUSED_FFN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFusedFFNDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFusedFFNDescriptor(
    infiniopHandle_t handle,
    infiniopFusedFFNDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t residual_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t gate_up_weight_desc,
    infiniopTensorDescriptor_t down_weight_desc,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetFusedFFNWorkspaceSize(
    infiniopFusedFFNDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopFusedFFN(
    infiniopFusedFFNDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *in,
    const void *residual,
    const void *norm_weight,
    const void *gate_up_weight,
    const void *down_weight,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFusedFFNDescriptor(
    infiniopFusedFFNDescriptor_t desc);

#endif
