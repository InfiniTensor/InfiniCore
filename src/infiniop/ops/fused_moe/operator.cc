#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/fused_moe.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/fused_moe_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateFusedMoeDescriptor(
    infiniopHandle_t handle,
    infiniopFusedMoeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t token_selected_experts_desc,
    infiniopTensorDescriptor_t token_final_scales_desc,
    infiniopTensorDescriptor_t w1_desc,
    infiniopTensorDescriptor_t w2_desc,
    infiniopTensorDescriptor_t b1_desc,
    infiniopTensorDescriptor_t b2_desc,
    infiniopFusedMoeActivation_t activation) {
#define CREATE(CASE, NAMESPACE)                                         \
    case CASE:                                                          \
        return op::fused_moe::NAMESPACE::Descriptor::create(            \
            handle, reinterpret_cast<op::fused_moe::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, input_desc, token_selected_experts_desc, token_final_scales_desc, \
            w1_desc, w2_desc, b1_desc, b2_desc, activation);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetFusedMoeWorkspaceSize(infiniopFusedMoeDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                               \
    case CASE:                                                                             \
        *size = reinterpret_cast<const op::fused_moe::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopFusedMoe(
    infiniopFusedMoeDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out,
    const void *input,
    const void *token_selected_experts,
    const void *token_final_scales,
    const void *w1,
    const void *w2,
    const void *b1,
    const void *b2,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                             \
        return reinterpret_cast<const op::fused_moe::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, input, token_selected_experts, token_final_scales, \
            w1, w2, b1, b2, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyFusedMoeDescriptor(infiniopFusedMoeDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                           \
    case CASE:                                                                             \
        delete reinterpret_cast<const op::fused_moe::NAMESPACE::Descriptor *>(desc);       \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
