#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/paged_attention_prefill.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/paged_attention_prefill_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreatePagedAttentionPrefillDescriptor(
    infiniopHandle_t handle,
    infiniopPagedAttentionPrefillDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t cache_lens_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t offset_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale) {

    infiniopTensorDescriptor_t alibi_opt = (alibi_slopes_desc == nullptr) ? nullptr : alibi_slopes_desc;

#define CREATE(CASE, NAMESPACE)                                                                \
    case CASE:                                                                                 \
        return op::paged_attention_prefill::NAMESPACE::Descriptor::create(                     \
            handle,                                                                            \
            reinterpret_cast<op::paged_attention_prefill::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc, cache_lens_desc,  \
            seq_lens_desc, offset_desc, alibi_opt, scale);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetPagedAttentionPrefillWorkspaceSize(
    infiniopPagedAttentionPrefillDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                                   \
    case CASE:                                                                                                 \
        *size = reinterpret_cast<op::paged_attention_prefill::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopPagedAttentionPrefill(
    infiniopPagedAttentionPrefillDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables, const void *cache_lens, const void *seq_lens,
    const void *offset,
    const void *alibi_slopes,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                          \
        return reinterpret_cast<op::paged_attention_prefill::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, q, k_cache, v_cache, block_tables,                          \
            cache_lens, seq_lens, offset, alibi_slopes, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyPagedAttentionPrefillDescriptor(
    infiniopPagedAttentionPrefillDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                             \
    case CASE:                                                                               \
        delete reinterpret_cast<op::paged_attention_prefill::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
