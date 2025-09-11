// infiniop/ops/chunk_gated_delta_rule/operator.cc

#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/chunk_gated_delta_rule.h"

#if defined(ENABLE_NVIDIA_API)
#include "nvidia/chunk_gated_delta_rule_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateChunkGatedDeltaRuleDescriptor(
    infiniopHandle_t handle,
    infiniopChunkGatedDeltaRuleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    bool use_qk_l2norm,
    size_t chunk_size
) {

    std::optional<infiniopTensorDescriptor_t> initial_state_opt = 
        (initial_state_desc == nullptr) ? std::nullopt : std::optional(initial_state_desc);

#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::chunk_gated_delta_rule::NAMESPACE::Descriptor::create(  \
            handle,                                                            \
            reinterpret_cast<                                                  \
                op::chunk_gated_delta_rule::NAMESPACE::Descriptor **>(     \
                desc_ptr),                                                     \
            out_desc, final_state_desc, q_desc, k_desc, v_desc, g_desc,         \
            beta_desc, initial_state_opt, use_qk_l2norm, chunk_size);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    // Add other devices like CPU, METAX here when available
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetChunkGatedDeltaRuleWorkspaceSize(
    infiniopChunkGatedDeltaRuleDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                   \
    case CASE:                                                                 \
        *size = reinterpret_cast<                                              \
            op::chunk_gated_delta_rule::NAMESPACE::Descriptor *>(desc)     \
            ->workspaceSize();                                                 \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopChunkGatedDeltaRule(
    infiniopChunkGatedDeltaRuleDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out, void* final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void* beta, const void* initial_state,
    void *stream
) {
#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<                                               \
            op::chunk_gated_delta_rule::NAMESPACE::Descriptor *>(desc)     \
            ->calculate(workspace, workspace_size, out, final_state, q, k, v,  \
                        g, beta, initial_state, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyChunkGatedDeltaRuleDescriptor(
    infiniopChunkGatedDeltaRuleDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                               \
    case CASE:                                                                 \
        delete reinterpret_cast<                                               \
            op::chunk_gated_delta_rule::NAMESPACE::Descriptor *>(desc);    \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}