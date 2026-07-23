#ifndef __INFINIOP_MOE_TOPK_SOFTMAX_API_H__
#define __INFINIOP_MOE_TOPK_SOFTMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoeTopkSoftmaxDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMoeTopkSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopMoeTopkSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t gating_output_desc,
    infiniopTensorDescriptor_t correction_bias_desc,
    bool renormalize,
    float moe_softcapping);

__INFINI_C __export infiniStatus_t infiniopGetMoeTopkSoftmaxWorkspaceSize(
    infiniopMoeTopkSoftmaxDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMoeTopkSoftmax(
    infiniopMoeTopkSoftmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *gating_output,
    const void *correction_bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMoeTopkSoftmaxDescriptor(
    infiniopMoeTopkSoftmaxDescriptor_t desc);

#endif
