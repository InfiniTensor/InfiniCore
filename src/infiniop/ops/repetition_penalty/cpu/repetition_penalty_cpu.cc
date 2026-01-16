#include "repetition_penalty_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../info.h"
#include "infinicore.h"
#include <algorithm>

namespace op::repetition_penalty::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t logits_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = RepetitionPenaltyInfo::create(logits_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,  // No workspace needed for CPU
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

template <typename T>
void apply_penalty_cpu(
    T *logits,
    const float *repetition_penalties,
    const uint32_t *token_indices,
    const size_t *token_offsets,
    size_t num_seqs,
    size_t vocab_size) {

    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        float penalty = repetition_penalties[seq_idx];
        if (penalty == 1.0f) {
            continue;  // Skip if no penalty
        }

        size_t start = token_offsets[seq_idx];
        size_t end = token_offsets[seq_idx + 1];
        for (size_t i = start; i < end; i++) {
            uint32_t token_id = token_indices[i];
            if (token_id >= vocab_size) {
                continue; // skip out-of-range ids
            }
            size_t offset = seq_idx * vocab_size + token_id;
            T logit_val_orig = logits[offset];
            float logit_val = utils::cast<float>(logit_val_orig);

            // Match PyTorch behavior exactly: val / p if val > 0 else val * p
            if (logit_val > 0.0f) {
                logits[offset] = utils::cast<T>(logit_val / penalty);
            } else {
                // For val <= 0: multiply by penalty (covers negative and zero)
                logits[offset] = utils::cast<T>(logit_val * penalty);
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *logits,
    const float *repetition_penalties,
    const uint32_t *token_indices,
    const size_t *token_offsets,
    size_t total_indices,
    void *stream) const {

    switch (_info.dt_logits) {
    case INFINI_DTYPE_F16:
        apply_penalty_cpu<fp16_t>(
            reinterpret_cast<fp16_t *>(logits),
            repetition_penalties,
            token_indices,
            token_offsets,
            _info.num_seqs,
            _info.vocab_size);
        break;
    case INFINI_DTYPE_BF16:
        apply_penalty_cpu<bf16_t>(
            reinterpret_cast<bf16_t *>(logits),
            repetition_penalties,
            token_indices,
            token_offsets,
            _info.num_seqs,
            _info.vocab_size);
        break;
    case INFINI_DTYPE_F32:
        apply_penalty_cpu<float>(
            reinterpret_cast<float *>(logits),
            repetition_penalties,
            token_indices,
            token_offsets,
            _info.num_seqs,
            _info.vocab_size);
        break;
    case INFINI_DTYPE_F64:
        apply_penalty_cpu<double>(
            reinterpret_cast<double *>(logits),
            repetition_penalties,
            token_indices,
            token_offsets,
            _info.num_seqs,
            _info.vocab_size);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::repetition_penalty::cpu
