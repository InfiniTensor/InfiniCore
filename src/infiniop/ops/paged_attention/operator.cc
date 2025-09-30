#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/paged_attention.h"

// Add necessary includes for different platforms
#ifdef ENABLE_CPU_API
// #include "cpu/paged_attention_cpu.h" // Placeholder for future CPU implementation
#endif
#if defined(ENABLE_NVIDIA_API)
#include "nvidia/paged_attention_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
// #include "metax/paged_attention_metax.h" // Placeholder
#endif
#ifdef ENABLE_ASCEND_API
// #include "ascend/paged_attention_ascend.h" // Placeholder
#endif

__C infiniStatus_t infiniopCreatePagedAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopPagedAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale
    ) {
    
    std::optional<infiniopTensorDescriptor_t> alibi_opt = 
        (alibi_slopes_desc == nullptr) ? std::nullopt : std::optional(alibi_slopes_desc);

#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::paged_attention::NAMESPACE::Descriptor::create(                     \
            handle,                                                                    \
            reinterpret_cast<op::paged_attention::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc, seq_lens_desc, alibi_opt, scale);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        // CREATE(INFINI_DEVICE_CPU, cpu) // Uncomment when CPU implementation is ready
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        // CREATE(INFINI_DEVICE_METAX, metax) // Placeholder for future Metax implementation
#endif
#ifdef ENABLE_ASCEND_API
        // CREATE(INFINI_DEVICE_ASCEND, ascend) // Placeholder for future Ascend implementation
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetPagedAttentionWorkspaceSize(
    infiniopPagedAttentionDescriptor_t desc, 
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                        \
        *size = reinterpret_cast<op::paged_attention::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        // GET(INFINI_DEVICE_CPU, cpu) // Uncomment when CPU implementation is ready
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        // GET(INFINI_DEVICE_METAX, metax) // Placeholder
#endif
#ifdef ENABLE_ASCEND_API
        // GET(INFINI_DEVICE_ASCEND, ascend) // Placeholder
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopPagedAttention(
    infiniopPagedAttentionDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables, const void *seq_lens, const void *alibi_slopes,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                      \
        return reinterpret_cast<op::paged_attention::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, q, k_cache, v_cache, block_tables,      \
            seq_lens, alibi_slopes, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        // CALCULATE(INFINI_DEVICE_CPU, cpu) // Uncomment when CPU implementation is ready
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        // CALCULATE(INFINI_DEVICE_METAX, metax) // Placeholder
#endif
#ifdef ENABLE_ASCEND_API
        // CALCULATE(INFINI_DEVICE_ASCEND, ascend) // Placeholder
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyPagedAttentionDescriptor(
    infiniopPagedAttentionDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                    \
    case CASE:                                                                      \
        delete reinterpret_cast<op::paged_attention::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        // DESTROY(INFINI_DEVICE_CPU, cpu) // Uncomment when CPU implementation is ready
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        // DESTROY(INFINI_DEVICE_METAX, metax) // Placeholder
#endif
#ifdef ENABLE_ASCEND_API
        // DESTROY(INFINI_DEVICE_ASCEND, ascend) // Placeholder
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}