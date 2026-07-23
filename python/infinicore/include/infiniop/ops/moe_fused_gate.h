#ifndef __INFINIOP_MOE_FUSED_GATE_API_H__
#define __INFINIOP_MOE_FUSED_GATE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoeFusedGateDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMoeFusedGateDescriptor(
    infiniopHandle_t handle,
    infiniopMoeFusedGateDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t bias_desc,
    size_t num_expert_group,
    size_t topk_group,
    size_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);

__INFINI_C __export infiniStatus_t infiniopGetMoeFusedGateWorkspaceSize(
    infiniopMoeFusedGateDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMoeFusedGate(
    infiniopMoeFusedGateDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *input,
    const void *bias,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMoeFusedGateDescriptor(
    infiniopMoeFusedGateDescriptor_t desc);

#endif
