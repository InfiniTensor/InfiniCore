#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/flash_attention.h"

#ifdef ENABLE_CPU_API
// #include "cpu/flash_attention_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_NVIDIA_API)
#include "ninetoothed/descriptor.h"
#else
// #include "nvidia/flash_attention_nvidia.cuh"
#endif
#endif

__C infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    float scale,
    char is_causal) {

#define CREATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        return op::flash_attention::NAMESPACE::Descriptor::create(                     \
            handle,                                                                    \
            reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc,                                                                  \
            q_desc,                                                                    \
            k_desc,                                                                    \
            v_desc,                                                                    \
            scale,                                                                     \
            is_causal);

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        // CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_NVIDIA_API)
        CREATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        // CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc,
    size_t *size) {

#define GET_SIZE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                             \
        *size = reinterpret_cast<const op::flash_attention::NAMESPACE::Descriptor *>(desc) \
                    ->get_workspace_size();                                                \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        // GET_SIZE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_NVIDIA_API)
        GET_SIZE(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        // GET_SIZE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET_SIZE
}

__C infiniStatus_t infiniopFlashAttention(
    infiniopFlashAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                            \
        return reinterpret_cast<const op::flash_attention::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, q, k, v, stream);

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        // CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_NVIDIA_API)
        CALCULATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        // CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyFlashAttentionDescriptor(
    infiniopFlashAttentionDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                     \
    case CASE:                                                                       \
        delete reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        // DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_NVIDIA_API)
        DESTROY(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        // DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
