// File: infiniop/ops/paged_caching/operator.cc

#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/paged_caching.h" // Assuming this is the public API header

// Add necessary includes for different platforms
#ifdef ENABLE_NVIDIA_API
#include "nvidia/paged_caching_nvidia.cuh"
#endif
// ... other platforms

__C infiniStatus_t infiniopCreatePagedCachingDescriptor(
    infiniopHandle_t handle,
    infiniopPagedCachingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {
    
#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::paged_caching::NAMESPACE::Descriptor::create(                       \
            handle,                                                                   \
            reinterpret_cast<op::paged_caching::NAMESPACE::Descriptor **>(desc_ptr),   \
            k_desc, v_desc, k_cache_desc, v_cache_desc, slot_mapping_desc);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    // ... other platforms
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetPagedCachingWorkspaceSize(
    infiniopPagedCachingDescriptor_t desc, 
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                        \
        *size = reinterpret_cast<op::paged_caching::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    // ... other platforms
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopPagedCaching(
    infiniopPagedCachingDescriptor_t desc,
    void *workspace, size_t workspace_size,
    const void *k, const void *v,
    void *k_cache, void *v_cache,
    const void *slot_mapping,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                      \
        return reinterpret_cast<op::paged_caching::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, k, v, k_cache, v_cache, slot_mapping, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    // ... other platforms请先用中文分析并阐述我的意图，再根据我的意图回答我的问题。
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyPagedCachingDescriptor(
    infiniopPagedCachingDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                    \
    case CASE:                                                                      \
        delete reinterpret_cast<op::paged_caching::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    // ... other platforms
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}