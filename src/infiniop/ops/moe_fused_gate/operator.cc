#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_fused_gate.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/moe_fused_gate_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateMoeFusedGateDescriptor(
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
    bool apply_routed_scaling_factor_on_output) {
#define CREATE(CASE, NAMESPACE)                                                               \
    case CASE:                                                                                \
        return op::moe_fused_gate::NAMESPACE::Descriptor::create(                             \
            handle, reinterpret_cast<op::moe_fused_gate::NAMESPACE::Descriptor **>(desc_ptr), \
            topk_weights_desc, topk_indices_desc, input_desc, bias_desc, num_expert_group,    \
            topk_group, num_fused_shared_experts, routed_scaling_factor,                      \
            apply_routed_scaling_factor_on_output)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMoeFusedGateWorkspaceSize(
    infiniopMoeFusedGateDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                        \
        *size = reinterpret_cast<op::moe_fused_gate::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopMoeFusedGate(
    infiniopMoeFusedGateDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *input,
    const void *bias,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                 \
        return reinterpret_cast<op::moe_fused_gate::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, topk_weights, topk_indices, input, bias, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyMoeFusedGateDescriptor(
    infiniopMoeFusedGateDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                    \
    case CASE:                                                                      \
        delete reinterpret_cast<op::moe_fused_gate::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
