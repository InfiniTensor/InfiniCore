/**
 * Vendor API declarations for InfLLM-v2 attention kernels.
 *
 * This header is intentionally placed under `infinicore/adaptor/` because it
 * declares symbols provided by an external InfLLM-v2 shared library.
 *
 * NOTE: The vendor functions are declared in the global namespace to match the
 * upstream InfLLM-v2 entrypoints (e.g. `entry.cu`) and to keep linkage stable.
 */
#pragma once

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <vector>

/** Varlen forward: unpadded Q/K/V with cu_seqlens. Returns {out, softmax_lse, ...}. */
std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    c10::optional<at::Tensor> &out_,
    const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k,
    c10::optional<at::Tensor> &seqused_k,
    c10::optional<const at::Tensor> &leftpad_k_,
    c10::optional<at::Tensor> &block_table_,
    c10::optional<at::Tensor> &alibi_slopes_,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool return_softmax,
    c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor> &blockmask_);

/** KV-cache forward (decode). Returns {out, softmax_lse}. */
std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor &q,
    const at::Tensor &kcache,
    const at::Tensor &vcache,
    c10::optional<const at::Tensor> &k_,
    c10::optional<const at::Tensor> &v_,
    c10::optional<const at::Tensor> &seqlens_k_,
    c10::optional<const at::Tensor> &rotary_cos_,
    c10::optional<const at::Tensor> &rotary_sin_,
    c10::optional<const at::Tensor> &cache_batch_idx_,
    c10::optional<const at::Tensor> &leftpad_k_,
    c10::optional<at::Tensor> &block_table_,
    c10::optional<at::Tensor> &alibi_slopes_,
    c10::optional<at::Tensor> &out_,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool is_rotary_interleaved,
    int num_splits,
    c10::optional<at::Tensor> &blockmask_);

#endif // ENABLE_INFLLMV2 && ENABLE_ATEN

