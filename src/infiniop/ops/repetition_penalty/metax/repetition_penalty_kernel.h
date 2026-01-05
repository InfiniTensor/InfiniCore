#ifndef __REPETITION_PENALTY_KERNEL_H__
#define __REPETITION_PENALTY_KERNEL_H__

#include "../../../devices/metax/metax_common.h"
#include "../info.h"

namespace op::repetition_penalty::metax {

// CUDA graph compatible kernel - all operations on device, no host-device memcpy
template <typename T>
static __global__ void applyRepetitionPenaltyKernel(
    T *__restrict__ logits,
    const bool *__restrict__ mask,
    const float *__restrict__ repetition_penalties,
    size_t num_seqs,
    size_t vocab_size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = num_seqs * vocab_size;

    if (idx >= total_elements) {
        return;
    }

    size_t seq_idx = idx / vocab_size;
    size_t token_idx = idx % vocab_size;

    float penalty = repetition_penalties[seq_idx];
    if (penalty == 1.0f) {
        return;  // No penalty, skip
    }

    if (mask[idx]) {
        float logit_val = static_cast<float>(logits[idx]);
        if (logit_val > 0) {
            logits[idx] = static_cast<T>(logit_val / penalty);
        } else {
            logits[idx] = static_cast<T>(logit_val * penalty);
        }
    }
}

} // namespace op::repetition_penalty::metax

#endif // __REPETITION_PENALTY_KERNEL_H__
