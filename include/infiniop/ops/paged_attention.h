#ifndef __INFINIOP_PAGED_ATTENTION_API_H__
#define __INFINIOP_PAGED_ATTENTION_API_H__

#include "../operator_descriptor.h"

// Define an opaque handle for the Paged Attention descriptor.
typedef struct InfiniopDescriptor *infiniopPagedAttentionDescriptor_t;

/**
 * @brief Creates a descriptor for the Paged Attention v1 operation.
 *
 * This function initializes a descriptor that holds all the metadata needed
 * for the paged attention computation.
 *
 * @param handle The handle to the InfiniOP library context.
 * @param desc_ptr A pointer to store the created descriptor.
 * @param out_desc Descriptor for the output tensor.
 * @param q_desc Descriptor for the query tensor.
 * @param k_cache_desc Descriptor for the key cache tensor.
 * @param v_cache_desc Descriptor for the value cache tensor.
 * @param block_tables_desc Descriptor for the block tables tensor.
 * @param seq_lens_desc Descriptor for the sequence lengths tensor.
 * @param alibi_slopes_desc Optional descriptor for the ALiBi slopes tensor. Can be NULL.
 * @param scale The attention scaling factor.
 * @param max_num_blocks_per_seq The maximum number of batched blocks tables.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopCreatePagedAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopPagedAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale);

/**
 * @brief Retrieves the workspace size required for the Paged Attention operation.
 *
 * @param desc The Paged Attention descriptor.
 * @param size A pointer to store the required workspace size in bytes.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopGetPagedAttentionWorkspaceSize(
    infiniopPagedAttentionDescriptor_t desc, size_t *size);

/**
 * @brief Executes the Paged Attention v1 operation.
 *
 * @param desc The Paged Attention descriptor.
 * @param workspace Pointer to the workspace memory.
 * @param workspace_size The size of the workspace.
 * @param out Pointer to the output tensor data.
 * @param q Pointer to the query tensor data.
 * @param k_cache Pointer to the key cache data.
 * @param v_cache Pointer to the value cache data.
 * @param block_tables Pointer to the block tables data.
 * @param seq_lens Pointer to the sequence lengths data.
 * @param alibi_slopes Pointer to the ALiBi slopes data. Can be NULL.
 * @param stream The CUDA stream for the operation. Can be NULL.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopPagedAttention(
    infiniopPagedAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *alibi_slopes,
    void *stream);

/**
 * @brief Destroys a Paged Attention descriptor.
 *
 * @param desc The descriptor to be destroyed.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopDestroyPagedAttentionDescriptor(
    infiniopPagedAttentionDescriptor_t desc);

#endif // __INFINIOP_PAGED_ATTENTION_API_H__