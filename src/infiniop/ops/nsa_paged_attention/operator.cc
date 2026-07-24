#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/nsa_paged_attention.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ALI_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
#include "nvidia/nsa_paged_attention_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/nsa_paged_attention_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateNsaPagedAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopNsaPagedAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cmp_desc,
    infiniopTensorDescriptor_t v_cmp_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t gates_desc,
    float scale,
    int nsa_block_size,
    int window_size,
    int select_blocks) {

#define CREATE(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                       \
        return op::nsa_paged_attention::NAMESPACE::Descriptor::create(                               \
            handle,                                                                                  \
            reinterpret_cast<op::nsa_paged_attention::NAMESPACE::Descriptor **>(desc_ptr),           \
            out_desc, q_desc, k_cmp_desc, v_cmp_desc, k_cache_desc, v_cache_desc, block_tables_desc, \
            seq_lens_desc, gates_desc, scale, nsa_block_size, window_size, select_blocks);

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

__INFINI_C infiniStatus_t infiniopGetNsaPagedAttentionWorkspaceSize(
    infiniopNsaPagedAttentionDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                               \
    case CASE:                                                                                             \
        *size = reinterpret_cast<op::nsa_paged_attention::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopNsaPagedAttention(
    infiniopNsaPagedAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cmp,
    const void *v_cmp,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *gates,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                      \
        return reinterpret_cast<op::nsa_paged_attention::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, seq_lens, gates, stream);

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

__INFINI_C infiniStatus_t infiniopDestroyNsaPagedAttentionDescriptor(
    infiniopNsaPagedAttentionDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                         \
    case CASE:                                                                           \
        delete reinterpret_cast<op::nsa_paged_attention::NAMESPACE::Descriptor *>(desc); \
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
