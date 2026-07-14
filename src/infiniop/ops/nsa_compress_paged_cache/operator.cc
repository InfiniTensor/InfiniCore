#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/nsa_compress_paged_cache.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ALI_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
#include "nvidia/nsa_compress_paged_cache_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/nsa_compress_paged_cache_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateNsaCompressPagedCacheDescriptor(
    infiniopHandle_t handle,
    infiniopNsaCompressPagedCacheDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t k_cmp_desc,
    infiniopTensorDescriptor_t v_cmp_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    int nsa_block_size,
    int update_last_only) {

#define CREATE(CASE, NAMESPACE)                                                                   \
    case CASE:                                                                                    \
        return op::nsa_compress_paged_cache::NAMESPACE::Descriptor::create(                       \
            handle,                                                                               \
            reinterpret_cast<op::nsa_compress_paged_cache::NAMESPACE::Descriptor **>(desc_ptr),   \
            k_cmp_desc, v_cmp_desc, k_cache_desc, v_cache_desc, block_tables_desc, seq_lens_desc, \
            nsa_block_size, update_last_only);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopGetNsaCompressPagedCacheWorkspaceSize(
    infiniopNsaCompressPagedCacheDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                    \
    case CASE:                                                                                                  \
        *size = reinterpret_cast<op::nsa_compress_paged_cache::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopNsaCompressPagedCache(
    infiniopNsaCompressPagedCacheDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *k_cmp,
    void *v_cmp,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                           \
        return reinterpret_cast<op::nsa_compress_paged_cache::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, k_cmp, v_cmp, k_cache, v_cache, block_tables, seq_lens, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopDestroyNsaCompressPagedCacheDescriptor(
    infiniopNsaCompressPagedCacheDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                              \
    case CASE:                                                                                \
        delete reinterpret_cast<op::nsa_compress_paged_cache::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
