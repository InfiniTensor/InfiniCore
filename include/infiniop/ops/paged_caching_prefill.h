#ifndef __INFINIOP_PAGED_CACHING_PREFILL_API_H__
#define __INFINIOP_PAGED_CACHING_PREFILL_API_H__

#include "../operator_descriptor.h"

// Define an opaque handle for the Paged Caching Prefill descriptor.
typedef struct InfiniopDescriptor *infiniopPagedCachingPrefillDescriptor_t;

/**
 * @brief Creates a descriptor for the Paged Caching Prefill operation.
 *
 * This function initializes a descriptor that holds metadata to copy key/value
 * vectors from a prefill batch into their respective physical slots in the cache pool.
 *
 * @param handle The handle to the InfiniOP library context.
 * @param desc_ptr A pointer to store the created descriptor.
 * @param k_desc Descriptor for the source key tensor (new tokens).
 * @param v_desc Descriptor for the source value tensor (new tokens).
 * @param k_cache_desc Descriptor for the key cache pool tensor (global pool).
 * @param v_cache_desc Descriptor for the value cache pool tensor (global pool).
 * @param slot_mapping_desc Descriptor for the slot mapping tensor (physical indices).
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopCreatePagedCachingPrefillDescriptor(
    infiniopHandle_t handle,
    infiniopPagedCachingPrefillDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc);

/**
 * @brief Retrieves the workspace size required for the Paged Caching Prefill operation.
 *
 * @param desc The Paged Caching Prefill descriptor.
 * @param size A pointer to store the required workspace size in bytes.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopGetPagedCachingPrefillWorkspaceSize(
    infiniopPagedCachingPrefillDescriptor_t desc, size_t *size);

/**
 * @brief Executes the Paged Caching Prefill operation.
 *
 * This operation writes the K/V data into the cache pool at locations 
 * specified by the slot_mapping.
 *
 * @param desc The Paged Caching Prefill descriptor.
 * @param workspace Pointer to the workspace memory.
 * @param workspace_size The size of the workspace.
 * @param k Pointer to the source key tensor data.
 * @param v Pointer to the source value tensor data.
 * @param k_cache Pointer to the key cache pool data.
 * @param v_cache Pointer to the value cache pool data.
 * @param slot_mapping Pointer to the slot mapping data.
 * @param stream The CUDA stream for the operation. Can be NULL.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopPagedCachingPrefill(
    infiniopPagedCachingPrefillDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *k,
    const void *v,
    void *k_cache,
    void *v_cache,
    const void *slot_mapping,
    void *stream);

/**
 * @brief Destroys a Paged Caching Prefill descriptor.
 *
 * @param desc The descriptor to be destroyed.
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopDestroyPagedCachingPrefillDescriptor(
    infiniopPagedCachingPrefillDescriptor_t desc);

#endif // __INFINIOP_PAGED_CACHING_PREFILL_API_H__
