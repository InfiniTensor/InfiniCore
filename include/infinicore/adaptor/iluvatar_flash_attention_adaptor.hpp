#pragma once

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ATen.h>

#include <optional>
#include <vector>

namespace pyinfer::cuinfer {

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    std::optional<const at::Tensor> &k_new,
    std::optional<const at::Tensor> &v_new,
    std::optional<const at::Tensor> &q_v,
    std::optional<at::Tensor> &out,
    std::optional<const at::Tensor> &cu_seqlens_q,
    std::optional<const at::Tensor> &cu_seqlens_k,
    std::optional<const at::Tensor> &cu_seqlens_k_new,
    std::optional<const at::Tensor> &seqused_q,
    std::optional<const at::Tensor> &seqused_k,
    std::optional<int> max_seqlen_q,
    std::optional<int> max_seqlen_k,
    std::optional<const at::Tensor> &block_table,
    std::optional<const at::Tensor> &kv_batch_idx,
    std::optional<const at::Tensor> &leftpad_k,
    std::optional<const at::Tensor> &rotary_cos,
    std::optional<const at::Tensor> &rotary_sin,
    std::optional<const at::Tensor> &seqlens_rotary,
    std::optional<at::Tensor> &q_descale,
    std::optional<at::Tensor> &k_descale,
    std::optional<at::Tensor> &v_descale,
    float softmax_scale,
    bool causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool rotary_interleaved,
    std::optional<at::Tensor> &scheduler_metadata,
    int num_splits,
    std::optional<bool> pack_gqa,
    int sm_margin,
    std::optional<const at::Tensor> &s_aux,
    int cp_world_size,
    int cp_rank,
    std::optional<const at::Tensor> &cp_tot_seqused_k);

} // namespace pyinfer::cuinfer
