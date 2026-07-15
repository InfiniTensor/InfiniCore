#pragma once

#include "infinicore/ops/inductor_segment.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#ifdef ENABLE_ATEN
#include <ATen/Tensor.h>
#endif

namespace infinicore::op::inductor_segment_impl {

/// Layer-agnostic AOT packages (one graph per bucket×tp_rank; weights bound at replay).
static constexpr size_t kLayerAgnosticIdx = static_cast<size_t>(-1);

struct SegmentKey {
    PiecewiseInductorSegmentId segment_id;
    size_t layer_idx;
    size_t bucket;
    size_t tp_rank;

    bool operator==(const SegmentKey &other) const {
        return segment_id == other.segment_id
               && layer_idx == other.layer_idx
               && bucket == other.bucket
               && tp_rank == other.tp_rank;
    }
};

struct SegmentKeyHash {
    size_t operator()(const SegmentKey &key) const {
        return (static_cast<size_t>(key.segment_id) << 40)
               ^ (key.layer_idx << 24)
               ^ (key.bucket << 8)
               ^ key.tp_rank;
    }
};

using TpRankResolver = size_t (*)();
using ValidSeqLenResolver = size_t (*)();

/// Thread-local override for lookup when TP global state is unavailable (smoke tests).
void set_lookup_tp_rank_override(size_t tp_rank);
void clear_lookup_tp_rank_override();
void set_tensor_parallel_rank_resolver(TpRankResolver resolver);
size_t current_tensor_parallel_rank();
void set_piecewise_valid_seq_len_resolver(ValidSeqLenResolver resolver);
size_t current_piecewise_valid_seq_len();
SegmentKey make_segment_key(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket);

#ifdef ENABLE_ATEN
class AotPackageRunner;

struct PreAttnExternalWeights {
    at::Tensor ln_weight;
    at::Tensor q_weight;
    at::Tensor k_weight;
    at::Tensor v_weight;
    at::Tensor q_norm_weight;
    at::Tensor k_norm_weight;
};

struct MoeExternalWeights {
    at::Tensor gate_weight;
    at::Tensor e_score_correction_bias;
    at::Tensor w_gate_up;
    at::Tensor w_down;
    at::Tensor shared_gate_up;
    at::Tensor shared_down;
};

using PreAttnWeightResolver = std::function<PreAttnExternalWeights(size_t layer_idx)>;
using MoeWeightResolver = std::function<MoeExternalWeights(size_t layer_idx)>;

void set_pre_attn_aten_weight_resolver(PreAttnWeightResolver resolver);
PreAttnExternalWeights resolve_pre_attn_weights(size_t layer_idx);

void set_moe_aten_weight_resolver(MoeWeightResolver resolver);
MoeExternalWeights resolve_moe_weights(size_t layer_idx);

void erase_inductor_runner(const SegmentKey &key);
void clear_inductor_runners();
#endif

class InductorSegmentRegistry {
public:
    static InductorSegmentRegistry &instance();

    void register_package(
        PiecewiseInductorSegmentId segment_id,
        size_t layer_idx,
        size_t bucket,
        const std::string &package_path,
        size_t tp_rank,
        bool layer_agnostic = false);

    void clear();

    bool has_package(
        PiecewiseInductorSegmentId segment_id,
        size_t layer_idx,
        size_t bucket,
        size_t tp_rank,
        bool layer_agnostic = false) const;

    bool package_is_layer_agnostic(
        PiecewiseInductorSegmentId segment_id,
        size_t layer_idx,
        size_t bucket,
        size_t tp_rank) const;

#ifdef ENABLE_ATEN
    AotPackageRunner &runner(
        PiecewiseInductorSegmentId segment_id,
        size_t layer_idx,
        size_t bucket,
        bool *layer_agnostic_out = nullptr);
#endif

private:
    InductorSegmentRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<SegmentKey, std::string, SegmentKeyHash> package_paths_;
    std::unordered_map<SegmentKey, bool, SegmentKeyHash> layer_agnostic_flags_;

friend AotPackageRunner &lookup_inductor_runner(
    InductorSegmentRegistry &registry,
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank,
    bool *layer_agnostic_out);
};

#ifdef ENABLE_ATEN
AotPackageRunner &lookup_inductor_runner(
    InductorSegmentRegistry &registry,
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank,
    bool *layer_agnostic_out = nullptr);
#endif

} // namespace infinicore::op::inductor_segment_impl
