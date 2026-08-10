#ifndef __GATED_DELTA_RULE_ASCEND_KERNEL_H__
#define __GATED_DELTA_RULE_ASCEND_KERNEL_H__

#include "../../../../../include/infinicore.h"
#include <cstddef>
#include <cstdint>

struct GatedDeltaRuleAscendParams {
    int32_t data_dtype;
    int32_t state_dtype;
    int32_t gate_dtype;
    bool use_qk_l2norm;
    bool has_cu_seqlens;
    bool cu_seqlens_i64;
    bool has_initial_indices;
    bool initial_indices_i64;
    bool has_final_indices;
    bool final_indices_i64;
    size_t B;
    size_t T;
    size_t total_tokens;
    size_t Hk;
    size_t Hv;
    size_t Dk;
    size_t Dv;
    size_t pool_size;
    size_t value_heads_per_key_head;
    float q_scale;
    ptrdiff_t out_strides[4];
    ptrdiff_t q_strides[4];
    ptrdiff_t k_strides[4];
    ptrdiff_t v_strides[4];
};

extern "C" infiniStatus_t gated_delta_rule_ascend_kernel_launch(
    void *workspace,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    const GatedDeltaRuleAscendParams *params,
    void *stream);

#endif
