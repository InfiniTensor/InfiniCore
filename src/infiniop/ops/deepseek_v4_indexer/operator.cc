#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/deepseek_v4_indexer.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/deepseek_v4_indexer_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDeepseekV4IndexerDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4IndexerDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t weights_desc,
    infiniopTensorDescriptor_t compressed_desc,
    infiniopTensorDescriptor_t positions_desc,
    size_t query_start,
    size_t compress_ratio) {
#define CREATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        return op::deepseek_v4_indexer::NAMESPACE::Descriptor::create(                   \
            handle, reinterpret_cast<op::deepseek_v4_indexer::NAMESPACE::Descriptor **>(desc_ptr), \
            indices_desc, q_desc, weights_desc, compressed_desc, positions_desc, query_start, compress_ratio)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4IndexerWorkspaceSize(
    infiniopDeepseekV4IndexerDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                           \
    case CASE:                                                                         \
        *size = reinterpret_cast<op::deepseek_v4_indexer::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopDeepseekV4Indexer(
    infiniopDeepseekV4IndexerDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *indices,
    const void *q,
    const void *weights,
    const void *compressed,
    const void *positions,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                         \
        return reinterpret_cast<op::deepseek_v4_indexer::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, indices, q, weights, compressed, positions, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4IndexerDescriptor(
    infiniopDeepseekV4IndexerDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                       \
    case CASE:                                                                         \
        delete reinterpret_cast<op::deepseek_v4_indexer::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
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
