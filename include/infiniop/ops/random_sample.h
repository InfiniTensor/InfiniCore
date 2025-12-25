#ifndef __INFINIOP_RANDOM_SAMPLE_API_H__
#define __INFINIOP_RANDOM_SAMPLE_API_H__

#include "../operator_descriptor.h"
#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopRandomSampleDescriptor_t;

__C __export infiniStatus_t infiniopCreateRandomSampleDescriptor(
    infiniopHandle_t handle,
    infiniopRandomSampleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t result,
    infiniopTensorDescriptor_t probs);

__C __export infiniStatus_t infiniopGetRandomSampleWorkspaceSize(
    infiniopRandomSampleDescriptor_t desc,
    size_t *size);

/**
 * @brief Performs random sampling with repetition penalty support.
 *
 * @param previous_tokens Array of UNIQUE token IDs that have appeared in the sequence.
 *                        Should contain no duplicates for optimal performance (vLLM-style).
 *                        Can be NULL if no tokens have been generated yet.
 *                        When NULL or previous_tokens_len is 0, falls back to full-history
 *                        penalty (applies penalty to all tokens) for backward compatibility.
 * @param previous_tokens_len Number of unique tokens in previous_tokens array.
 *                            Must be 0 if previous_tokens is NULL.
 *
 * @note For best performance, pass only unique token IDs (no duplicates).
 *       The implementation applies penalty only to tokens in this array.
 *       This follows vLLM's efficient approach: O(U) instead of O(T) where
 *       U = unique tokens << T = total tokens.
 */
__C __export infiniStatus_t infiniopRandomSample(
    infiniopRandomSampleDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    float repetition_penalty,
    const uint32_t *previous_tokens,  // Array of unique previously generated token IDs
    size_t previous_tokens_len,       // Number of unique tokens (0 if NULL)
    void *stream);

__C __export infiniStatus_t infiniopDestroyRandomSampleDescriptor(
    infiniopRandomSampleDescriptor_t desc);

#endif
