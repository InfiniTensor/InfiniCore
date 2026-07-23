#ifndef __INFINIOP_MOE_TOPK_SIGMOID_API_H__
#define __INFINIOP_MOE_TOPK_SIGMOID_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoeTopkSigmoidDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMoeTopkSigmoidDescriptor(
    infiniopHandle_t handle,
    infiniopMoeTopkSigmoidDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t gating_output_desc,
    infiniopTensorDescriptor_t correction_bias_desc,
    bool renormalize);

__INFINI_C __export infiniStatus_t infiniopGetMoeTopkSigmoidWorkspaceSize(
    infiniopMoeTopkSigmoidDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMoeTopkSigmoid(
    infiniopMoeTopkSigmoidDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *gating_output,
    const void *correction_bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMoeTopkSigmoidDescriptor(
    infiniopMoeTopkSigmoidDescriptor_t desc);

#endif
