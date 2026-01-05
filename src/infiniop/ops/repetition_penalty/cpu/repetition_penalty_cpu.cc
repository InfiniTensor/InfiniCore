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
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t mask_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = RepetitionPenaltyInfo::create(logits_desc, mask_desc);
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
    const bool *mask,
    const float *repetition_penalties,
    size_t num_seqs,
    size_t vocab_size) {

    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        float penalty = repetition_penalties[seq_idx];
        if (penalty == 1.0f) {
            continue;  // Skip if no penalty
        }

        size_t logits_offset = seq_idx * vocab_size;
        size_t mask_offset = seq_idx * vocab_size;

        for (size_t token_idx = 0; token_idx < vocab_size; token_idx++) {
            if (mask[mask_offset + token_idx]) {
                float logit_val = static_cast<float>(logits[logits_offset + token_idx]);
                if (logit_val > 0) {
                    logits[logits_offset + token_idx] = static_cast<T>(logit_val / penalty);
                } else {
                    logits[logits_offset + token_idx] = static_cast<T>(logit_val * penalty);
                }
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *logits,
    const void *mask,
    const float *repetition_penalties,
    void *stream) const {

    switch (_info.dt_logits) {
    case INFINI_DTYPE_F16:
        apply_penalty_cpu<fp16_t>(
            reinterpret_cast<fp16_t *>(logits),
            reinterpret_cast<const bool *>(mask),
            repetition_penalties,
            _info.num_seqs,
            _info.vocab_size);
        break;
    case INFINI_DTYPE_BF16:
        apply_penalty_cpu<bf16_t>(
            reinterpret_cast<bf16_t *>(logits),
            reinterpret_cast<const bool *>(mask),
            repetition_penalties,
            _info.num_seqs,
            _info.vocab_size);
        break;
    case INFINI_DTYPE_F32:
        apply_penalty_cpu<float>(
            reinterpret_cast<float *>(logits),
            reinterpret_cast<const bool *>(mask),
            repetition_penalties,
            _info.num_seqs,
            _info.vocab_size);
        break;
    case INFINI_DTYPE_F64:
        apply_penalty_cpu<double>(
            reinterpret_cast<double *>(logits),
            reinterpret_cast<const bool *>(mask),
            repetition_penalties,
            _info.num_seqs,
            _info.vocab_size);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::repetition_penalty::cpu
