#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/simple_gla_attention.h"
#include "infiniop/ops/simple_gla_prefill.h"

#ifdef ENABLE_CPU_API
#include "cpu/simple_gla_attention_cpu.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateSimpleGLAAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopSimpleGLAAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_gamma_desc) {

#define CREATE_CPU(CASE)                                                              \
    case CASE:                                                                        \
        return op::simple_gla_attention::cpu::Descriptor::create(                     \
            handle,                                                                   \
            reinterpret_cast<op::simple_gla_attention::cpu::Descriptor **>(desc_ptr), \
            out_desc, q_desc, k_desc, v_desc, g_gamma_desc)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE_CPU(INFINI_DEVICE_CPU);
#endif
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return infiniopCreateSimpleGLAPrefillDescriptor(
            handle,
            reinterpret_cast<infiniopSimpleGLAPrefillDescriptor_t *>(desc_ptr),
            out_desc,
            q_desc,
            k_desc,
            v_desc,
            g_gamma_desc);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE_CPU
}

__INFINI_C infiniStatus_t infiniopGetSimpleGLAAttentionWorkspaceSize(
    infiniopSimpleGLAAttentionDescriptor_t desc,
    size_t *size) {

#define WS_CPU(CASE)                                                                                  \
    case CASE:                                                                                        \
        *size = reinterpret_cast<op::simple_gla_attention::cpu::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        WS_CPU(INFINI_DEVICE_CPU);
#endif
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return infiniopGetSimpleGLAPrefillWorkspaceSize(
            reinterpret_cast<infiniopSimpleGLAPrefillDescriptor_t>(desc), size);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef WS_CPU
}

__INFINI_C infiniStatus_t infiniopSimpleGLAAttention(
    infiniopSimpleGLAAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void const *q,
    void const *k,
    void const *v,
    void const *g_gamma,
    float scale,
    void *stream) {

#define CALC_CPU(CASE)                                                             \
    case CASE:                                                                     \
        return reinterpret_cast<op::simple_gla_attention::cpu::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, q, k, v, g_gamma, scale, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALC_CPU(INFINI_DEVICE_CPU);
#endif
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return infiniopSimpleGLAPrefill(
            reinterpret_cast<infiniopSimpleGLAPrefillDescriptor_t>(desc),
            workspace,
            workspace_size,
            out,
            q,
            k,
            v,
            g_gamma,
            scale,
            stream);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALC_CPU
}

__INFINI_C infiniStatus_t infiniopDestroySimpleGLAAttentionDescriptor(
    infiniopSimpleGLAAttentionDescriptor_t desc) {

#define DESTROY_CPU(CASE)                                                           \
    case CASE:                                                                      \
        delete reinterpret_cast<op::simple_gla_attention::cpu::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DESTROY_CPU(INFINI_DEVICE_CPU);
#endif
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return infiniopDestroySimpleGLAPrefillDescriptor(
            reinterpret_cast<infiniopSimpleGLAPrefillDescriptor_t>(desc));
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY_CPU
}
