#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/kimi_delta_attention.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
#include "nvidia/kimi_delta_attention_nvidia.cuh"
#endif

__INFINI_C __export infiniStatus_t infiniopCreateKimiDeltaAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopKimiDeltaAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t A_log_desc,
    infiniopTensorDescriptor_t dt_bias_desc,
    infiniopTensorDescriptor_t cu_seqlens_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc,
    float scale,
    float lower_bound,
    bool use_qk_l2norm) {

#define CREATE(CASE, NAMESPACE)                                                               \
    case CASE:                                                                                \
        return op::kimi_delta_attention::NAMESPACE::Descriptor::create(                       \
            handle, reinterpret_cast<op::kimi_delta_attention::NAMESPACE::Descriptor **>(     \
                        desc_ptr),                                                           \
            out_desc, initial_state_desc, final_state_desc, q_desc, k_desc, v_desc, g_desc,   \
            beta_desc, A_log_desc, dt_bias_desc, cu_seqlens_desc, initial_state_indices_desc, \
            final_state_indices_desc, scale, lower_bound, use_qk_l2norm)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C __export infiniStatus_t infiniopGetKimiDeltaAttentionWorkspaceSize(
    infiniopKimiDeltaAttentionDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<op::kimi_delta_attention::NAMESPACE::Descriptor *>(desc)        \
                    ->workspaceSize();                                                           \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C __export infiniStatus_t infiniopKimiDeltaAttention(
    infiniopKimiDeltaAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *A_log,
    const void *dt_bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                             \
        return reinterpret_cast<const op::kimi_delta_attention::NAMESPACE::Descriptor *>(  \
            desc)                                                                          \
            ->calculate(workspace, workspace_size, out, initial_state, final_state, q, k,  \
                        v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices,     \
                        final_state_indices, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C __export infiniStatus_t infiniopDestroyKimiDeltaAttentionDescriptor(
    infiniopKimiDeltaAttentionDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                               \
    case CASE:                                                                                \
        delete reinterpret_cast<const op::kimi_delta_attention::NAMESPACE::Descriptor *>(     \
            desc);                                                                            \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DELETE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
