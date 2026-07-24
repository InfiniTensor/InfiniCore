#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dsv4_swa_prefill_indices.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/dsv4_swa_prefill_indices_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateDsv4SwaPrefillIndicesDescriptor(infiniopHandle_t handle, infiniopDsv4SwaPrefillIndicesDescriptor_t *desc_ptr, infiniopTensorDescriptor_t indices_desc, int window_size) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::dsv4_swa_prefill_indices::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::dsv4_swa_prefill_indices::NAMESPACE::Descriptor **>(desc_ptr), indices_desc, window_size)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}
__INFINI_C infiniStatus_t infiniopGetDsv4SwaPrefillIndicesWorkspaceSize(infiniopDsv4SwaPrefillIndicesDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                    \
    case CASE:                                                                                                  \
        *size = reinterpret_cast<op::dsv4_swa_prefill_indices::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}
__INFINI_C infiniStatus_t infiniopDsv4SwaPrefillIndices(infiniopDsv4SwaPrefillIndicesDescriptor_t desc, void *workspace, size_t workspace_size, void *indices, void *stream) {
#define CALC(CASE, NAMESPACE) \
    case CASE:                \
        return reinterpret_cast<op::dsv4_swa_prefill_indices::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, indices, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALC(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALC(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALC(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALC(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALC(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALC
}
__INFINI_C infiniStatus_t infiniopDestroyDsv4SwaPrefillIndicesDescriptor(infiniopDsv4SwaPrefillIndicesDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                              \
    case CASE:                                                                                \
        delete reinterpret_cast<op::dsv4_swa_prefill_indices::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
