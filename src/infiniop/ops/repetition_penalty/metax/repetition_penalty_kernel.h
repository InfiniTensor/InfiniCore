#ifndef __REPETITION_PENALTY_KERNEL_H__
#define __REPETITION_PENALTY_KERNEL_H__

#include "../../../devices/metax/metax_common.h"
#include "../info.h"

namespace op::repetition_penalty::metax {

// CUDA graph compatible kernel - all operations on device, no host-device memcpy
template <typename T>
static __global__ void applyRepetitionPenaltyKernel(
    T *__restrict__ logits,
    const float *__restrict__ repetition_penalties,
    const uint32_t *__restrict__ token_indices,
    const size_t *__restrict__ token_offsets,
    size_t num_seqs,
    size_t vocab_size,
    size_t total_indices) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_indices) {
        return;
    }

    // Binary search over token_offsets to find seq_idx such that
    // token_offsets[seq_idx] <= idx < token_offsets[seq_idx + 1]
    size_t lo = 0;
    size_t hi = num_seqs;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (token_offsets[mid + 1] <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    size_t seq_idx = lo;

    uint32_t token_id = token_indices[idx];
    if (token_id >= vocab_size) {
        return;
    }

    float penalty = repetition_penalties[seq_idx];
    if (penalty == 1.0f) {
        return;  // No penalty, skip
    }

    size_t offset = seq_idx * vocab_size + token_id;
    float logit_val = static_cast<float>(logits[offset]);
    if (logit_val > 0) {
        logits[offset] = static_cast<T>(logit_val / penalty);
    } else {
        logits[offset] = static_cast<T>(logit_val * penalty);
    }
}

} // namespace op::repetition_penalty::metax

#endif // __REPETITION_PENALTY_KERNEL_H__
