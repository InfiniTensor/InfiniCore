// infiniop/ops/recurrent_gated_delta_rule/operator.cc

#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/recurrent_gated_delta_rule.h"

#if defined(ENABLE_NVIDIA_API)
#include "nvidia/recurrent_gated_delta_rule_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateRecurrentGatedDeltaRuleDescriptor(
    infiniopHandle_t handle,
    infiniopRecurrentGatedDeltaRuleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    bool use_qk_l2norm
) {
#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::recurrent_gated_delta_rule::NAMESPACE::Descriptor::create(  \
            handle,                                                            \
            reinterpret_cast<                                                  \
                op::recurrent_gated_delta_rule::NAMESPACE::Descriptor **>(     \
                desc_ptr),                                                     \
            out_desc, final_state_desc, q_desc, k_desc, v_desc, g_desc,         \
            beta_desc, initial_state_desc, use_qk_l2norm);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    // Add other devices like CPU, METAX here when available
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetRecurrentGatedDeltaRuleWorkspaceSize(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                   \
    case CASE:                                                                 \
        *size = reinterpret_cast<                                              \
            op::recurrent_gated_delta_rule::NAMESPACE::Descriptor *>(desc)     \
            ->workspaceSize();                                                 \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopRecurrentGatedDeltaRule(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out, void* final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void* beta, const void* initial_state,
    void *stream
) {
#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<                                               \
            op::recurrent_gated_delta_rule::NAMESPACE::Descriptor *>(desc)     \
            ->calculate(workspace, workspace_size, out, final_state, q, k, v,  \
                        g, beta, initial_state, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyRecurrentGatedDeltaRuleDescriptor(
    infiniopRecurrentGatedDeltaRuleDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                               \
    case CASE:                                                                 \
        delete reinterpret_cast<                                               \
            op::recurrent_gated_delta_rule::NAMESPACE::Descriptor *>(desc);    \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}