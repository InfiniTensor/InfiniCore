// infiniop/ops/lightning_attention/operator.cc

#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/lightning_attention.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/lightning_attention_nvidia.cuh"
#endif
#ifdef ENABLE_CPU_API
#include "cpu/lightning_attention_cpu.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateLightningAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopLightningAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slope_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc) {
#define CREATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                     \
        return op::lightning_attention::NAMESPACE::Descriptor::create(             \
            handle,                                                                \
            reinterpret_cast<op::lightning_attention::NAMESPACE::Descriptor **>(   \
                desc_ptr),                                                         \
            out_desc, initial_state_desc, q_desc, k_desc, v_desc, slope_desc,      \
            initial_state_indices_desc, final_state_indices_desc);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetLightningAttentionWorkspaceSize(
    infiniopLightningAttentionDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                              \
    case CASE:                                                            \
        *size = reinterpret_cast<                                         \
                    op::lightning_attention::NAMESPACE::Descriptor *>(    \
                    desc)                                                 \
                    ->workspaceSize();                                    \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopLightningAttention(
    infiniopLightningAttentionDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out, void *initial_state,
    const void *q, const void *k, const void *v,
    const void *slope,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                    \
        return reinterpret_cast<                                                  \
                   op::lightning_attention::NAMESPACE::Descriptor *>(desc)        \
            ->calculate(workspace, workspace_size, out, initial_state,            \
                        q, k, v, slope, initial_state_indices,                    \
                        final_state_indices, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyLightningAttentionDescriptor(
    infiniopLightningAttentionDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                            \
    case CASE:                                                              \
        delete reinterpret_cast<                                            \
            op::lightning_attention::NAMESPACE::Descriptor *>(desc);        \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
