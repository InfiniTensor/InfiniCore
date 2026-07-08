#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/// M4 piecewise segment ids (mirror ``piecewise_segments.py``).
enum class PiecewiseInductorSegmentId : int {
    PreAttn = 0,
    PostAttnCg = 1,
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

namespace inductor_segment_impl {

void register_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    const std::string &package_path,
    size_t tp_rank = 0);

void clear_packages();

bool has_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket);

void set_lookup_tp_rank_override(size_t tp_rank);
void clear_lookup_tp_rank_override();
void set_tensor_parallel_rank_resolver(size_t (*)());

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
