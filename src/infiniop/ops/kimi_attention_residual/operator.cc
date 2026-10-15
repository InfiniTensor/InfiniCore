#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/kimi_attention_residual.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/kimi_attention_residual_nvidia.cuh"
#endif

__INFINI_C __export infiniStatus_t infiniopCreateKimiAttentionResidualDescriptor(
    infiniopHandle_t handle,
    infiniopKimiAttentionResidualDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t projection_weight_desc,
    float epsilon) {
#define CREATE(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                   \
        return op::kimi_attention_residual::NAMESPACE::Descriptor::create(                       \
            handle,                                                                              \
            reinterpret_cast<op::kimi_attention_residual::NAMESPACE::Descriptor **>(desc_ptr),   \
            output_desc, values_desc, norm_weight_desc, projection_weight_desc, epsilon)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C __export infiniStatus_t infiniopGetKimiAttentionResidualWorkspaceSize(
    infiniopKimiAttentionResidualDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                    \
    case CASE:                                                                                  \
        *size = reinterpret_cast<op::kimi_attention_residual::NAMESPACE::Descriptor *>(desc)    \
                    ->workspaceSize();                                                          \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C __export infiniStatus_t infiniopKimiAttentionResidual(
    infiniopKimiAttentionResidualDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *values,
    const void *norm_weight,
    const void *projection_weight,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                                  \
        return reinterpret_cast<const op::kimi_attention_residual::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, values, norm_weight, projection_weight, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C __export infiniStatus_t infiniopDestroyKimiAttentionResidualDescriptor(
    infiniopKimiAttentionResidualDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                  \
        delete reinterpret_cast<const op::kimi_attention_residual::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}
