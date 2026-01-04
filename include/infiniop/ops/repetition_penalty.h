#ifndef __INFINIOP_REPETITION_PENALTY_API_H__
#define __INFINIOP_REPETITION_PENALTY_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRepetitionPenaltyDescriptor_t;

/**
 * @brief Creates a descriptor for the Repetition Penalty operation.
 *
 * CUDA graph-compatible: descriptor depends only on fixed tensor shapes.
 *
 * @param handle The handle to the InfiniOP library context.
 * @param desc_ptr A pointer to store the created descriptor.
 * @param logits_desc Descriptor for logits tensor [num_seqs, vocab_size]
 * @param mask_desc Descriptor for boolean mask tensor [num_seqs, vocab_size]
 * @return infiniStatus_t Status code of the operation.
 */
__C __export infiniStatus_t infiniopCreateRepetitionPenaltyDescriptor(
    infiniopHandle_t handle,
    infiniopRepetitionPenaltyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t mask_desc);

__C __export infiniStatus_t infiniopGetRepetitionPenaltyWorkspaceSize(
    infiniopRepetitionPenaltyDescriptor_t desc,
    size_t *size);

/**
 * @brief Applies repetition penalty to logits in-place (CUDA graph compatible).
 *
 * Uses boolean mask pattern similar to vLLM for CUDA graph compatibility.
 * The mask indicates which tokens have appeared (True = penalize, False = no penalty).
 *
 * @param desc The Repetition Penalty descriptor (cached based on tensor shapes only).
 * @param workspace Pointer to workspace memory.
 * @param workspace_size Size of workspace in bytes.
 * @param logits Logits tensor [num_seqs, vocab_size] - modified in-place (device pointer)
 * @param mask Boolean mask tensor [num_seqs, vocab_size] - True for tokens to penalize (device pointer)
 * @param repetition_penalties Penalty values [num_seqs] - one per sequence (device pointer for GPU backends, host pointer for CPU)
 *                              For CUDA graph compatibility on GPU backends, this must be a device pointer.
 *                              The caller must copy penalties to device before calling this function.
 * @param stream CUDA stream (can be NULL)
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopApplyRepetitionPenalty(
    infiniopRepetitionPenaltyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *logits,                    // [num_seqs, vocab_size] - in-place modification (device pointer)
    const void *mask,                // [num_seqs, vocab_size] - bool tensor (device pointer)
    const float *repetition_penalties, // [num_seqs] - device pointer for GPU, host pointer for CPU
    void *stream);

__C __export infiniStatus_t infiniopDestroyRepetitionPenaltyDescriptor(
    infiniopRepetitionPenaltyDescriptor_t desc);

#endif
