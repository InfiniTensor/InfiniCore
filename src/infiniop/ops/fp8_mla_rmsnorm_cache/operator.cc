#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/fp8_mla_rmsnorm_cache.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API)
#include "nvidia/fp8_mla_rmsnorm_cache_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateFp8MlaRmsnormCacheDescriptor(
    infiniopHandle_t handle,
    infiniopFp8MlaRmsnormCacheDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t cache_desc,
    infiniopTensorDescriptor_t vendor_cache_desc,
    infiniopTensorDescriptor_t compressed_kv_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t rope_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    double eps) {
#define CREATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                               \
        return op::fp8_mla_rmsnorm_cache::NAMESPACE::Descriptor::create(                     \
            handle,                                                                          \
            reinterpret_cast<op::fp8_mla_rmsnorm_cache::NAMESPACE::Descriptor **>(desc_ptr), \
            cache_desc, vendor_cache_desc, compressed_kv_desc,                               \
            norm_weight_desc, rope_desc, slot_mapping_desc, eps)
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

__INFINI_C infiniStatus_t infiniopFp8MlaRmsnormCache(
    infiniopFp8MlaRmsnormCacheDescriptor_t desc,
    void *cache,
    void *vendor_cache,
    const void *compressed_kv,
    const void *norm_weight,
    const void *rope,
    const void *slot_mapping,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                                  \
        return reinterpret_cast<const op::fp8_mla_rmsnorm_cache::NAMESPACE::Descriptor *>(desc) \
            ->calculate(cache, vendor_cache, compressed_kv, norm_weight,                        \
                        rope, slot_mapping, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyFp8MlaRmsnormCacheDescriptor(
    infiniopFp8MlaRmsnormCacheDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                   \
        delete reinterpret_cast<const op::fp8_mla_rmsnorm_cache::NAMESPACE::Descriptor *>(desc); \
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
