#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/flash_attention_backward.h"

#ifdef ENABLE_CPU_API
#include "cpu/flash_attention_backward_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/flash_attention_backward_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/flash_attention_backward_metax.cuh"
#endif

__C infiniStatus_t infiniopCreateFlashAttentionBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_q_desc,
    infiniopTensorDescriptor_t grad_k_desc,
    infiniopTensorDescriptor_t grad_v_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t grad_out_desc,
    infiniopTensorDescriptor_t mask_desc,
    infiniopAttentionMaskType_t mask_type) {

#define CREATE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                  \
        return op::flash_attention_backward::NAMESPACE::Descriptor::create(                     \
            handle,                                                                             \
            reinterpret_cast<op::flash_attention_backward::NAMESPACE::Descriptor **>(desc_ptr), \
            grad_q_desc,                                                                        \
            grad_k_desc,                                                                        \
            grad_v_desc,                                                                        \
            q_desc,                                                                             \
            k_desc,                                                                             \
            v_desc,                                                                             \
            grad_out_desc,                                                                      \
            mask_desc,                                                                          \
            mask_type);

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetFlashAttentionBackwardWorkspaceSize(
    infiniopFlashAttentionBackwardDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                \
        *size = reinterpret_cast<op::flash_attention_backward::NAMESPACE::Descriptor *>(desc) \
                    ->workspaceSize();                                                        \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniStatus_t infiniopFlashAttentionBackward(
    infiniopFlashAttentionBackwardDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *grad_q,
    void *grad_k,
    void *grad_v,
    const void *q,
    const void *k,
    const void *v,
    const void *grad_out,
    const void *mask,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                               \
        return reinterpret_cast<op::flash_attention_backward::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                           \
                        grad_q, grad_k, grad_v, q, k, v, grad_out, mask, stream);

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyFlashAttentionBackwardDescriptor(
    infiniopFlashAttentionBackwardDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                              \
    case CASE:                                                                                \
        delete reinterpret_cast<op::flash_attention_backward::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
