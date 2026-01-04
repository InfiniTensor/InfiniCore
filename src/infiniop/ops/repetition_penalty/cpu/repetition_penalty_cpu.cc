#include "repetition_penalty_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../info.h"
#include "infinicore.h"
#include <cstring>

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

template <typename T>
infiniStatus_t applyRepetitionPenalty(
    void *logits,
    const void *mask,
    const float *repetition_penalties,
    size_t num_seqs,
    size_t vocab_size,
    void *stream) {

    T *logits_ptr = reinterpret_cast<T *>(logits);
    const bool *mask_ptr = reinterpret_cast<const bool *>(mask);

    #pragma omp parallel for collapse(2)
    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        for (size_t vocab_idx = 0; vocab_idx < vocab_size; vocab_idx++) {
            size_t idx = seq_idx * vocab_size + vocab_idx;

            if (mask_ptr[idx]) {
                float penalty = repetition_penalties[seq_idx];
                // Use utils::cast to properly handle fp16_t and bf16_t conversions
                float logit_val = utils::cast<float>(logits_ptr[idx]);

                if (logit_val > 0) {
                    logits_ptr[idx] = utils::cast<T>(logit_val / penalty);
                } else {
                    logits_ptr[idx] = utils::cast<T>(logit_val * penalty);
                }
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *logits,
    const void *mask,
    const float *repetition_penalties,
    void *stream) const {

    size_t num_seqs = _info.num_seqs;
    size_t vocab_size = _info.vocab_size;

    switch (_info.dt_logits) {
        case INFINI_DTYPE_F16:
            return applyRepetitionPenalty<fp16_t>(
                logits, mask, repetition_penalties, num_seqs, vocab_size, stream);
        case INFINI_DTYPE_F32:
            return applyRepetitionPenalty<float>(
                logits, mask, repetition_penalties, num_seqs, vocab_size, stream);
        case INFINI_DTYPE_BF16:
            return applyRepetitionPenalty<bf16_t>(
                logits, mask, repetition_penalties, num_seqs, vocab_size, stream);
        case INFINI_DTYPE_F64:
            return applyRepetitionPenalty<double>(
                logits, mask, repetition_penalties, num_seqs, vocab_size, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::repetition_penalty::cpu
