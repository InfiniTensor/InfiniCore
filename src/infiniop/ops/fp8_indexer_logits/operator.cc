#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/fp8_indexer_logits.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API)
#include "nvidia/fp8_indexer_logits_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateFp8IndexerLogitsDescriptor(
    infiniopHandle_t handle,
    infiniopFp8IndexerLogitsDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t positions_desc,
    infiniopTensorDescriptor_t request_ids_desc) {
#define CREATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                            \
        return op::fp8_indexer_logits::NAMESPACE::Descriptor::create(                     \
            handle,                                                                       \
            reinterpret_cast<op::fp8_indexer_logits::NAMESPACE::Descriptor **>(desc_ptr), \
            logits_desc, q_fp8_desc, kv_cache_desc, block_tables_desc,                    \
            weights_fp32_desc, positions_desc, request_ids_desc)
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
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopFp8IndexerLogits(
    infiniopFp8IndexerLogitsDescriptor_t desc,
    void *logits,
    const void *q_fp8,
    const void *kv_cache,
    const void *block_tables,
    const void *weights_fp32,
    const void *positions,
    const void *request_ids,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                               \
        return reinterpret_cast<const op::fp8_indexer_logits::NAMESPACE::Descriptor *>(desc) \
            ->calculate(logits, q_fp8, kv_cache, block_tables, weights_fp32,                 \
                        positions, request_ids, stream)
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
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyFp8IndexerLogitsDescriptor(
    infiniopFp8IndexerLogitsDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                              \
    case CASE:                                                                                \
        delete reinterpret_cast<const op::fp8_indexer_logits::NAMESPACE::Descriptor *>(desc); \
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
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
