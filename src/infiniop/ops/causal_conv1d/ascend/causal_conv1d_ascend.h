#ifndef __CAUSAL_CONV1D_ASCEND_H__
#define __CAUSAL_CONV1D_ASCEND_H__
#include "../causal_conv1d.h"
DESCRIPTOR(ascend)
namespace op::causal_conv1d::ascend {
extern "C" infiniStatus_t causal_conv1d_kernel_launch(
    void *out, void *conv_state, void *final_conv_state,
    const void *qkv, const void *weight, const void *bias,
    const void *cu_seqlens, const void *initial_state_indices,
    const void *final_state_indices, infiniDtype_t dtype,
    bool has_bias, bool has_cu_seqlens, bool cu_seqlens_i64,
    bool initial_state_indices_i64, bool final_state_indices_i64,
    bool indexed_state_pool, size_t request_count, size_t T,
    size_t C, size_t total_tokens, size_t pool_size,
    ptrdiff_t out_s0, ptrdiff_t out_s1, ptrdiff_t out_s2,
    ptrdiff_t state_s0, ptrdiff_t state_s1, ptrdiff_t state_s2,
    ptrdiff_t final_s0, ptrdiff_t final_s1, ptrdiff_t final_s2,
    ptrdiff_t qkv_s0, ptrdiff_t qkv_s1, ptrdiff_t qkv_s2,
    ptrdiff_t weight_s0, ptrdiff_t weight_s2, ptrdiff_t bias_s0,
    void *stream);
}
#endif
