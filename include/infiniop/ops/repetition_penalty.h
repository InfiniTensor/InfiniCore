#ifndef __INFINIOP_REPETITION_PENALTY_API_H__
#define __INFINIOP_REPETITION_PENALTY_API_H__

#include "../operator_descriptor.h"
#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopRepetitionPenaltyDescriptor_t;

/**
 * @brief Creates a repetition penalty operator descriptor.
 *
 * @param handle InfiniCore handle
 * @param desc_ptr Output descriptor pointer
 * @param logits_desc Logits tensor descriptor [num_seqs, vocab_size] - will be modified in-place
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopCreateRepetitionPenaltyDescriptor(
    infiniopHandle_t handle,
    infiniopRepetitionPenaltyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t logits_desc);

/**
 * @brief Gets the workspace size required for repetition penalty operation.
 *
 * @param desc Operator descriptor
 * @param size Output workspace size
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopGetRepetitionPenaltyWorkspaceSize(
    infiniopRepetitionPenaltyDescriptor_t desc,
    size_t *size);

/**
 * @brief Applies repetition penalty to logits in-place using token indices only.
 *
 * @param desc Operator descriptor
 * @param workspace Workspace buffer
 * @param workspace_size Workspace size
 * @param logits Logits tensor [num_seqs, vocab_size] - modified in-place (device pointer)
 * @param repetition_penalties Repetition penalty values [num_seqs] - device pointer for GPU backends, host pointer for CPU
 * @param token_indices Flattened token ids to penalize (device pointer)
 * @param token_offsets Prefix sums into token_indices, length = num_seqs + 1 (device pointer)
 * @param total_indices Total number of token indices across all sequences (token_offsets[num_seqs])
 * @param stream CUDA stream
 * @return infiniStatus_t Status code
 *
 * @note For CUDA graph compatibility:
 *       - repetition_penalties and token buffers must be device pointers for GPU backends
 *       - total_indices must be computed on host before graph capture
 *       - The caller is responsible for copying penalty values and token buffers to device before graph capture
 */
__C __export infiniStatus_t infiniopApplyRepetitionPenalty(
    infiniopRepetitionPenaltyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *logits,
    const float *repetition_penalties,
    const uint32_t *token_indices,   // flattened token ids to penalize
    const size_t *token_offsets,     // prefix sum, len = num_seqs + 1
    size_t total_indices,            // total number of indices (token_offsets[num_seqs])
    void *stream);

/**
 * @brief Destroys a repetition penalty operator descriptor.
 *
 * @param desc Operator descriptor
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopDestroyRepetitionPenaltyDescriptor(
    infiniopRepetitionPenaltyDescriptor_t desc);

#endif
