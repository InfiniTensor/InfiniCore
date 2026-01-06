#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/kv_caching.h"

#ifdef ENABLE_CPU_API
// #include "cpu/kv_caching_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
// #include "nvidia/kv_caching_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateKVCachingDescriptor(
    infiniopHandle_t handle,
    infiniopKVCachingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t k_cache,
    infiniopTensorDescriptor_t v_cache,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t v,
    infiniopTensorDescriptor_t past_kv_lengths) {

#define CREATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::kv_caching::NAMESPACE::Descriptor::create(                     \
            handle,                                                               \
            reinterpret_cast<op::kv_caching::NAMESPACE::Descriptor **>(desc_ptr), \
            k_cache,                                                              \
            v_cache,                                                              \
            k,                                                                    \
            v,                                                                    \
            past_kv_lengths)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        // CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        // CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetKVCachingWorkspaceSize(
    infiniopKVCachingDescriptor_t desc,
    size_t *size) {

#define GET_SIZE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                        \
        *size = reinterpret_cast<const op::kv_caching::NAMESPACE::Descriptor *>(desc) \
                    ->get_workspace_size();                                           \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        // GET_SIZE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        // GET_SIZE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET_SIZE
}

__C infiniStatus_t infiniopKVCaching(
    infiniopKVCachingDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *k_cache,
    void *v_cache,
    const void *k,
    const void *v,
    const void *past_kv_lengths,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                       \
        return reinterpret_cast<const op::kv_caching::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, k_cache, v_cache, k, v, past_kv_lengths, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        // CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        // CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyKVCachingDescriptor(
    infiniopKVCachingDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<op::kv_caching::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        // DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        // DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}
