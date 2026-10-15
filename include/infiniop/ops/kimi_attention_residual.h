#ifndef __INFINIOP_KIMI_ATTENTION_RESIDUAL_API_H__
#define __INFINIOP_KIMI_ATTENTION_RESIDUAL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopKimiAttentionResidualDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateKimiAttentionResidualDescriptor(
    infiniopHandle_t handle,
    infiniopKimiAttentionResidualDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t projection_weight_desc,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetKimiAttentionResidualWorkspaceSize(
    infiniopKimiAttentionResidualDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopKimiAttentionResidual(
    infiniopKimiAttentionResidualDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *values,
    const void *norm_weight,
    const void *projection_weight,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyKimiAttentionResidualDescriptor(
    infiniopKimiAttentionResidualDescriptor_t desc);

#endif
