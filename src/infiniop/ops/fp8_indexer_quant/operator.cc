#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/fp8_indexer_quant.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API)
#include "nvidia/fp8_indexer_quant_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateFp8IndexerQuantDescriptor(
    infiniopHandle_t handle,
    infiniopFp8IndexerQuantDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t weights_desc) {
#define CREATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        return op::fp8_indexer_quant::NAMESPACE::Descriptor::create(                     \
            handle,                                                                      \
            reinterpret_cast<op::fp8_indexer_quant::NAMESPACE::Descriptor **>(desc_ptr), \
            q_fp8_desc, weights_fp32_desc, q_desc, weights_desc)
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

__INFINI_C infiniStatus_t infiniopFp8IndexerQuant(
    infiniopFp8IndexerQuantDescriptor_t desc,
    void *q_fp8,
    void *weights_fp32,
    const void *q,
    const void *weights,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                              \
        return reinterpret_cast<const op::fp8_indexer_quant::NAMESPACE::Descriptor *>(desc) \
            ->calculate(q_fp8, weights_fp32, q, weights, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyFp8IndexerQuantDescriptor(
    infiniopFp8IndexerQuantDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                             \
    case CASE:                                                                               \
        delete reinterpret_cast<const op::fp8_indexer_quant::NAMESPACE::Descriptor *>(desc); \
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

__INFINI_C infiniStatus_t infiniopCreateFusedFp8IndexerDescriptor(
    infiniopHandle_t handle,
    infiniopFusedFp8IndexerDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t q_raw_desc,
    infiniopTensorDescriptor_t k_weights_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t norm_bias_desc,
    infiniopTensorDescriptor_t positions_desc,
    infiniopTensorDescriptor_t cos_sin_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    uint64_t rope_dim,
    double eps,
    double weights_scale) {
#define CREATE_FUSED(CASE, NAMESPACE)                                                         \
    case CASE:                                                                                \
        return op::fp8_indexer_quant::NAMESPACE::FusedDescriptor::create(                     \
            handle,                                                                           \
            reinterpret_cast<op::fp8_indexer_quant::NAMESPACE::FusedDescriptor **>(desc_ptr), \
            q_fp8_desc, weights_fp32_desc, k_cache_desc, q_raw_desc,                          \
            k_weights_desc, norm_weight_desc, norm_bias_desc, positions_desc,                 \
            cos_sin_cache_desc, slot_mapping_desc, rope_dim, eps, weights_scale)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE_FUSED(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE_FUSED(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE_FUSED(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE_FUSED(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE_FUSED
}

__INFINI_C infiniStatus_t infiniopFusedFp8Indexer(
    infiniopFusedFp8IndexerDescriptor_t desc,
    void *q_fp8,
    void *weights_fp32,
    void *k_cache,
    const void *q_raw,
    const void *k_weights,
    const void *norm_weight,
    const void *norm_bias,
    const void *positions,
    const void *cos_sin_cache,
    const void *slot_mapping,
    void *stream) {
#define CALCULATE_FUSED(CASE, NAMESPACE)                                                         \
    case CASE:                                                                                   \
        return reinterpret_cast<const op::fp8_indexer_quant::NAMESPACE::FusedDescriptor *>(desc) \
            ->calculate(q_fp8, weights_fp32, k_cache, q_raw, k_weights,                          \
                        norm_weight, norm_bias, positions, cos_sin_cache,                        \
                        slot_mapping, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE_FUSED(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE_FUSED(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE_FUSED(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE_FUSED(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE_FUSED
}

__INFINI_C infiniStatus_t infiniopDestroyFusedFp8IndexerDescriptor(
    infiniopFusedFp8IndexerDescriptor_t desc) {
#define DESTROY_FUSED(CASE, NAMESPACE)                                                            \
    case CASE:                                                                                    \
        delete reinterpret_cast<const op::fp8_indexer_quant::NAMESPACE::FusedDescriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY_FUSED(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY_FUSED(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY_FUSED(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY_FUSED(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY_FUSED
}
