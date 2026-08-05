#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/mrope.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/mrope_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/mrope_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/mrope_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateMRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopMRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_out,
    infiniopTensorDescriptor_t k_out,
    infiniopTensorDescriptor_t q,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t cos,
    infiniopTensorDescriptor_t sin,
    infiniopTensorDescriptor_t positions,
    int head_size,
    int rotary_dim,
    int section_t,
    int section_h,
    int section_w,
    bool interleaved) {

#define CREATE(CASE, NAMESPACE)                                              \
    case CASE:                                                               \
        return op::mrope::NAMESPACE::Descriptor::create(                     \
            handle,                                                          \
            reinterpret_cast<op::mrope::NAMESPACE::Descriptor **>(desc_ptr), \
            q_out, k_out, q, k, cos, sin, positions, head_size, rotary_dim,  \
            section_t, section_h, section_w, interleaved)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMRoPEWorkspaceSize(infiniopMRoPEDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<const op::mrope::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopMRoPE(
    infiniopMRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *q_out,
    void *k_out,
    const void *q,
    const void *k,
    const void *cos,
    const void *sin,
    const void *positions,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<const op::mrope::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, q_out, k_out, q, k, cos, sin, positions, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyMRoPEDescriptor(infiniopMRoPEDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                           \
    case CASE:                                                             \
        delete reinterpret_cast<op::mrope::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
