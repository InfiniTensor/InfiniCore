#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <functional>
#include <string>

namespace infinicore::op {

/// M4 piecewise segment ids (mirror ``piecewise_segments.py``).
enum class PiecewiseInductorSegmentId : int {
    PreAttn = 0,
    PostAttnCg = 1,
    Moe = 2,
};

INFINICORE_GRAPH_OP_CLASS(
    InductorSegment,
    const Tensor &,
    Tensor &,
    Tensor &,
    Tensor &,
    Tensor &,
    Tensor &,
    PiecewiseInductorSegmentId,
    size_t,
    size_t);

/// Run an AOTInductor piecewise segment inside hcGraph capture/replay.
void inductor_segment_(
    const Tensor &positions,
    Tensor &hidden_states,
    Tensor &residual,
    Tensor &q_rope,
    Tensor &k_rope,
    Tensor &v_rope,
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket);

/// Eager AOT warmup on the current rank/device (before hcGraph capture).
void inductor_warmup_pre_attn_bucket(
    const Tensor &positions,
    Tensor &positions_padded,
    Tensor &hidden_states,
    Tensor &residual,
    size_t layer_idx,
    size_t bucket,
    size_t valid_len);

/// Eager AOTI MiniCPM5 sparse MoE segment (weights via moe resolver). Not recorded into hcGraph yet.
void inductor_moe_(
    const Tensor &hidden_states,
    Tensor &out,
    size_t layer_idx,
    size_t bucket);

namespace inductor_segment_impl {

struct PreAttnExternalWeightTensors {
    Tensor ln_weight;
    Tensor q_weight;
    Tensor k_weight;
    Tensor v_weight;
    Tensor q_norm_weight;
    Tensor k_norm_weight;
};

/// Layer-agnostic MoE external weights (bound at replay via resolver).
struct MoeExternalWeightTensors {
    Tensor gate_weight;              ///< [E, H]
    Tensor e_score_correction_bias;  ///< [E]
    Tensor w_gate_up;                ///< [E, 2*I, H]
    Tensor w_down;                   ///< [E, H, I]
    Tensor shared_gate_up;           ///< [2*I, H]
    Tensor shared_down;              ///< [H, I]
};

void register_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    const std::string &package_path,
    size_t tp_rank = 0,
    bool layer_agnostic = false);

void clear_packages();

bool has_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket);

void set_pre_attn_weight_resolver(
    std::function<PreAttnExternalWeightTensors(size_t layer_idx)> resolver);

void clear_pre_attn_weight_resolver();

void set_moe_weight_resolver(
    std::function<MoeExternalWeightTensors(size_t layer_idx)> resolver);

void clear_moe_weight_resolver();

/// True after a non-null ``set_moe_weight_resolver`` (RankWorker or pybind).
bool has_moe_weight_resolver();

void set_lookup_tp_rank_override(size_t tp_rank);
void clear_lookup_tp_rank_override();
void set_tensor_parallel_rank_resolver(size_t (*)());
void set_piecewise_valid_seq_len_resolver(size_t (*)());

#ifdef ENABLE_ATEN
void warmup_pre_attn(
    const infinicore::Tensor &positions,
    infinicore::Tensor &positions_padded,
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &residual,
    size_t layer_idx,
    size_t bucket,
    size_t valid_len);
#endif

} // namespace inductor_segment_impl

} // namespace infinicore::op
