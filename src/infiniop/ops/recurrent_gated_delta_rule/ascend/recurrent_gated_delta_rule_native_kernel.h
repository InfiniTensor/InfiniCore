#ifndef __RECURRENT_GATED_DELTA_RULE_NATIVE_KERNEL_H__
#define __RECURRENT_GATED_DELTA_RULE_NATIVE_KERNEL_H__

#include "../../../../../include/infinicore.h"
#include <cstddef>
#include <cstdint>

struct RecurrentGdrNativeParams {
    size_t B;
    size_t Hk;
    size_t Hv;
    size_t pool_size;
    bool initial_indices_i64;
    bool final_indices_i64;
    ptrdiff_t q_s0;
    ptrdiff_t q_s2;
    ptrdiff_t k_s0;
    ptrdiff_t k_s2;
    ptrdiff_t v_s0;
    ptrdiff_t v_s2;
    ptrdiff_t beta_s0;
    ptrdiff_t beta_s2;
};

extern "C" infiniStatus_t recurrent_gdr_native_preprocess_launch(
    void *q_normalized,
    void *k_normalized,
    void *v_contiguous,
    void *beta_bf16,
    void *state_staging,
    void *actual_seq_lengths,
    void *state_indices,
    const void *q,
    const void *k,
    const void *v,
    const void *beta,
    const void *state,
    const void *initial_state_indices,
    const void *final_state_indices,
    const RecurrentGdrNativeParams *params,
    void *stream);

extern "C" infiniStatus_t recurrent_gdr_native_commit_state_launch(
    void *state,
    const void *state_staging,
    const void *initial_state_indices,
    const void *final_state_indices,
    const RecurrentGdrNativeParams *params,
    void *stream);

#endif
