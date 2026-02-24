#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/paged_attention_v1.h"

#if defined(ENABLE_NVIDIA_API)
#include "nvidia/paged_attention_v1_nvidia.cuh"
#endif
#if defined(ENABLE_METAX_API)
#include "metax/paged_attention_v1_metax.cuh"
#endif
__C infiniStatus_t infiniopCreatePagedAttentionV1Descriptor(
    infiniopHandle_t handle,
    infiniopPagedAttentionV1Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t key_cache_desc,
    infiniopTensorDescriptor_t value_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    double scale) {

#define CREATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                            \
        return op::paged_attention_v1::NAMESPACE::Descriptor::create(                     \
            handle,                                                                       \
            reinterpret_cast<op::paged_attention_v1::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, query_desc, key_cache_desc, value_cache_desc, block_tables_desc, seq_lens_desc, alibi_slopes_desc, scale);

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

__C infiniStatus_t infiniopGetPagedAttentionV1WorkspaceSize(
    infiniopPagedAttentionV1Descriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                              \
    case CASE:                                                                                            \
        *size = reinterpret_cast<op::paged_attention_v1::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__C infiniStatus_t infiniopPagedAttentionV1(
    infiniopPagedAttentionV1Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,            // [num_seqs, num_heads, head_size]
    void *query,          // [num_seqs, num_heads, head_size]
    void *key_cache,      // [num_blocks, num_heads, head_size/x, block_size, x]
    void *value_cache,    // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, // [num_heads]
    double scale,
    void *block_tables, // [num_seqs, max_num_blocks_per_seq]
    void *seq_lens,     // [num_seqs]
    int64_t block_size,
    int64_t max_seq_len,
    const void *alibi_slopes,
    const char *kv_cache_dtype,
    void *k_scale,
    void *v_scale,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                                                                                                                              \
    case CASE:                                                                                                                                                                                                  \
        return reinterpret_cast<op::paged_attention_v1::NAMESPACE::Descriptor *>(desc)->calculate(                                                                                                              \
            workspace, workspace_size, out, query, key_cache, value_cache, num_kv_heads,                                                                                                                        \
            scale, static_cast<int64_t *>(block_tables), static_cast<int64_t *>(seq_lens), block_size, max_seq_len, alibi_slopes, kv_cache_dtype, static_cast<float *>(k_scale), static_cast<float *>(v_scale), \
            tp_rank, blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size, blocksparse_head_sliding_step,                                                                                  \
            stream);

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

__C infiniStatus_t infiniopDestroyPagedAttentionV1Descriptor(
    infiniopPagedAttentionV1Descriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                        \
    case CASE:                                                                          \
        delete reinterpret_cast<op::paged_attention_v1::NAMESPACE::Descriptor *>(desc); \
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
