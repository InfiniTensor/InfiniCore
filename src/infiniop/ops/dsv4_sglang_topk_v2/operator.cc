#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dsv4_sglang_topk_v2.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/dsv4_sglang_topk_v2_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDsv4SglangTopkV2Descriptor(infiniopHandle_t handle, infiniopDsv4SglangTopkV2Descriptor_t *desc_ptr, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_table_desc, infiniopTensorDescriptor_t page_indices_desc, infiniopTensorDescriptor_t transform_workspace_desc, infiniopTensorDescriptor_t metadata_desc, int64_t page_size) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::dsv4_sglang_topk_v2::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::dsv4_sglang_topk_v2::NAMESPACE::Descriptor **>(desc_ptr), scores_desc, seq_lens_desc, page_table_desc, page_indices_desc, transform_workspace_desc, metadata_desc, page_size)
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

__INFINI_C infiniStatus_t infiniopGetDsv4SglangTopkV2WorkspaceSize(infiniopDsv4SglangTopkV2Descriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                               \
    case CASE:                                                                                             \
        *size = reinterpret_cast<op::dsv4_sglang_topk_v2::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopDsv4SglangTopkV2(infiniopDsv4SglangTopkV2Descriptor_t desc, void *workspace, size_t workspace_size, const void *scores, const void *seq_lens, const void *page_table, void *page_indices, void *transform_workspace, void *metadata, void *stream) {
#define CALCULATE(CASE, NAMESPACE) \
    case CASE:                     \
        return reinterpret_cast<op::dsv4_sglang_topk_v2::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, scores, seq_lens, page_table, page_indices, transform_workspace, metadata, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyDsv4SglangTopkV2Descriptor(infiniopDsv4SglangTopkV2Descriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                         \
    case CASE:                                                                           \
        delete reinterpret_cast<op::dsv4_sglang_topk_v2::NAMESPACE::Descriptor *>(desc); \
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
