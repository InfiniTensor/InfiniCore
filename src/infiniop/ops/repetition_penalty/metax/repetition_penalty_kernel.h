#ifndef __REPETITION_PENALTY_METAX_KERNEL_H__
#define __REPETITION_PENALTY_METAX_KERNEL_H__

#include "../../../devices/metax/metax_kernel_common.h"

namespace op::repetition_penalty::metax {

template <unsigned int BLOCK_SIZE, typename T>
INFINIOP_METAX_KERNEL applyRepetitionPenaltyKernel(
    T *__restrict__ logits,
    const bool *__restrict__ mask,
    const float *__restrict__ repetition_penalties,
    size_t num_seqs,
    size_t vocab_size) {

    // Each thread processes one vocab element
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);

    if (idx >= num_seqs * vocab_size) {
        return;
    }

    size_t seq_idx = idx / vocab_size;
    size_t vocab_idx = idx % vocab_size;

    // Bounds check for seq_idx
    if (seq_idx >= num_seqs) {
        return;
    }

    if (mask[idx]) {
        float penalty = repetition_penalties[seq_idx];

        // Convert to float for computation
        float logit_val = static_cast<float>(logits[idx]);

        // Apply penalty
        float result;
        if (logit_val > 0) {
            result = logit_val / penalty;
        } else {
            result = logit_val * penalty;
        }

        // Convert back to T
        logits[idx] = static_cast<T>(result);
    }
}

template <unsigned int BLOCK_SIZE, typename T>
void launchKernel(
    T *logits,
    const bool *mask,
    const float *repetition_penalties,
    size_t num_seqs,
    size_t vocab_size,
    hcStream_t stream) {

    size_t total_elements = num_seqs * vocab_size;
    size_t grid_size = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(static_cast<uint32_t>(grid_size));
    dim3 block(BLOCK_SIZE);
    size_t shared_mem_size = 0;

    applyRepetitionPenaltyKernel<BLOCK_SIZE, T>
        <<<grid, block, shared_mem_size, stream>>>(
            logits, mask, repetition_penalties, num_seqs, vocab_size);
}

} // namespace op::repetition_penalty::metax

#endif // __REPETITION_PENALTY_METAX_KERNEL_H__
