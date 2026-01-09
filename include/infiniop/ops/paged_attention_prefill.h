#ifndef __INFINIOP_PAGED_ATTENTION_PREFILL_API_H__
#define __INFINIOP_PAGED_ATTENTION_PREFILL_API_H__

#include "../operator_descriptor.h"

// Define an opaque handle for the Paged Attention Prefill descriptor.
typedef struct InfiniopDescriptor *infiniopPagedAttentionPrefillDescriptor_t;

/**
 * @brief Creates a descriptor for the Paged Attention Prefill operation.
 * @param handle The handle to the InfiniOP library context.
 * @param desc_ptr A pointer to store the created descriptor.
 * @param out_desc Descriptor for the output tensor.
 * @param q_desc Descriptor for the query tensor (packed/flattened).
 * @param k_cache_desc Descriptor for the global physical key cache.
 * @param v_cache_desc Descriptor for the global physical value cache.
 * @param block_tables_desc Descriptor for the block tables mapping logic to physical blocks.
 * @param cache_lens_desc Descriptor for the total sequence lengths (history + current).
 * @param seq_lens_desc Descriptor for the current prefill sequence lengths.
 * @param offset_desc Descriptor for the start position of each sequence in the packed Q tensor.
 * @param alibi_slopes_desc Optional descriptor for the ALiBi slopes tensor. Can be NULL.
 * @param scale The attention scaling factor.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopCreatePagedAttentionPrefillDescriptor(
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
    float scale);

/**
 * @brief Retrieves the workspace size required for the Paged Attention Prefill operation.
 */
__C __export infiniStatus_t infiniopGetPagedAttentionPrefillWorkspaceSize(
    infiniopPagedAttentionPrefillDescriptor_t desc, size_t *size);

/**
 * @brief Executes the Paged Attention Prefill operation.
 * @param desc The Paged Attention Prefill descriptor.
 * @param workspace Pointer to the workspace memory.
 * @param workspace_size The size of the workspace.
 * @param out Pointer to the output tensor data.
 * @param q Pointer to the query tensor data (packed).
 * @param k_cache Pointer to the global key cache data.
 * @param v_cache Pointer to the global value cache data.
 * @param block_tables Pointer to the block tables data.
 * @param cache_lens Pointer to the total sequence lengths data.
 * @param seq_lens Pointer to the current prefill sequence lengths data.
 * @param offset Pointer to the sequence start offsets data.
 * @param alibi_slopes Pointer to the ALiBi slopes data. Can be NULL.
 * @param stream The CUDA/device stream for the operation.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopPagedAttentionPrefill(
    infiniopPagedAttentionPrefillDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *cache_lens,
    const void *seq_lens,
    const void *offset,
    const void *alibi_slopes,
    void *stream);

/**
 * @brief Destroys a Paged Attention Prefill descriptor.
 */
__C __export infiniStatus_t infiniopDestroyPagedAttentionPrefillDescriptor(
    infiniopPagedAttentionPrefillDescriptor_t desc);

#endif // __INFINIOP_PAGED_ATTENTION_PREFILL_API_H__
