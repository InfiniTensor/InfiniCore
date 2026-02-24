#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/reshape_and_cache.h"

#if defined(ENABLE_NVIDIA_API)
#include "nvidia/reshape_and_cache_nvidia.cuh"
#endif
#if defined(ENABLE_METAX_API)
#include "metax/reshape_and_cache_metax.h"
#endif

__C infiniStatus_t infiniopCreateReshapeAndCacheDescriptor(
    infiniopHandle_t handle,
    infiniopReshapeAndCacheDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t key_desc,
    infiniopTensorDescriptor_t value_desc,
    infiniopTensorDescriptor_t key_cache_desc,
    infiniopTensorDescriptor_t value_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    const char *kv_cache_dtype) {

#define CREATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        return op::reshape_and_cache::NAMESPACE::Descriptor::create(                     \
            handle,                                                                      \
            reinterpret_cast<op::reshape_and_cache::NAMESPACE::Descriptor **>(desc_ptr), \
            key_desc, value_desc, key_cache_desc, value_cache_desc, slot_mapping_desc, kv_cache_dtype);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetReshapeAndCacheWorkspaceSize(
    infiniopReshapeAndCacheDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                             \
    case CASE:                                                                                           \
        *size = reinterpret_cast<op::reshape_and_cache::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopReshapeAndCache(
    infiniopReshapeAndCacheDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *key,
    void *value,
    void *key_cache,
    void *value_cache,
    const void *slot_mapping,
    const char *kv_cache_dtype,
    void *k_scale,
    void *v_scale,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                \
    case CASE:                                                                                    \
        return reinterpret_cast<op::reshape_and_cache::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyReshapeAndCacheDescriptor(
    infiniopReshapeAndCacheDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                       \
    case CASE:                                                                         \
        delete reinterpret_cast<op::reshape_and_cache::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
