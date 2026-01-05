#ifndef __INFINIOP_REPETITION_PENALTY_API_H__
#define __INFINIOP_REPETITION_PENALTY_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRepetitionPenaltyDescriptor_t;

/**
 * @brief Creates a repetition penalty operator descriptor.
 *
 * @param handle InfiniCore handle
 * @param desc_ptr Output descriptor pointer
 * @param logits_desc Logits tensor descriptor [num_seqs, vocab_size] - will be modified in-place
 * @param mask_desc Boolean mask tensor descriptor [num_seqs, vocab_size] - true for tokens to penalize
 * @return infiniStatus_t Status code
 */
__C __export infiniStatus_t infiniopCreateRepetitionPenaltyDescriptor(
    infiniopHandle_t handle,
    infiniopRepetitionPenaltyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t mask_desc);

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
 * @brief Applies repetition penalty to logits in-place.
 *
 * This operator modifies logits based on a boolean mask and repetition penalty values.
 * For tokens where mask is true, if logit > 0: logit = logit / penalty, else: logit = logit * penalty.
 *
 * @param desc Operator descriptor
 * @param workspace Workspace buffer
 * @param workspace_size Workspace size
 * @param logits Logits tensor [num_seqs, vocab_size] - modified in-place (device pointer)
 * @param mask Boolean mask tensor [num_seqs, vocab_size] - true for tokens to penalize (device pointer)
 * @param repetition_penalties Repetition penalty values [num_seqs] - device pointer for GPU backends, host pointer for CPU
 * @param stream CUDA stream
 * @return infiniStatus_t Status code
 *
 * @note For CUDA graph compatibility, repetition_penalties must be a device pointer for GPU backends.
 *       The caller is responsible for copying penalty values to device before graph capture.
 */
__C __export infiniStatus_t infiniopApplyRepetitionPenalty(
    infiniopRepetitionPenaltyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *logits,
    const void *mask,
    const float *repetition_penalties,
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
