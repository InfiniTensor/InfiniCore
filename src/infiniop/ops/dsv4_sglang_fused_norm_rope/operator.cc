#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dsv4_sglang_fused_norm_rope.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/dsv4_sglang_fused_norm_rope_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDsv4SglangFusedNormRopeDescriptor(infiniopHandle_t handle, infiniopDsv4SglangFusedNormRopeDescriptor_t *desc_ptr, infiniopTensorDescriptor_t kv_desc, infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t positions_desc, infiniopTensorDescriptor_t freqs_desc, double eps) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::dsv4_sglang_fused_norm_rope::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::dsv4_sglang_fused_norm_rope::NAMESPACE::Descriptor **>(desc_ptr), kv_desc, weight_desc, positions_desc, freqs_desc, eps)
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

__INFINI_C infiniStatus_t infiniopGetDsv4SglangFusedNormRopeWorkspaceSize(infiniopDsv4SglangFusedNormRopeDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                       \
    case CASE:                                                                                                     \
        *size = reinterpret_cast<op::dsv4_sglang_fused_norm_rope::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopDsv4SglangFusedNormRope(infiniopDsv4SglangFusedNormRopeDescriptor_t desc, void *workspace, size_t workspace_size, void *kv, const void *weight, const void *positions, const void *freqs, void *stream) {
#define CALCULATE(CASE, NAMESPACE) \
    case CASE:                     \
        return reinterpret_cast<op::dsv4_sglang_fused_norm_rope::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, kv, weight, positions, freqs, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyDsv4SglangFusedNormRopeDescriptor(infiniopDsv4SglangFusedNormRopeDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                   \
        delete reinterpret_cast<op::dsv4_sglang_fused_norm_rope::NAMESPACE::Descriptor *>(desc); \
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
