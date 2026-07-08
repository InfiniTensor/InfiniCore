#pragma once

#include "infinicore/ops/inductor_segment.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace infinicore::op::inductor_segment_impl {

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

/// Thread-local override for lookup when TP global state is unavailable (smoke tests).
void set_lookup_tp_rank_override(size_t tp_rank);
void clear_lookup_tp_rank_override();
void set_tensor_parallel_rank_resolver(TpRankResolver resolver);
size_t current_tensor_parallel_rank();
SegmentKey make_segment_key(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket);

#ifdef ENABLE_ATEN
class AotPackageRunner;

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
        size_t tp_rank);

    void clear();

    bool has_package(
        PiecewiseInductorSegmentId segment_id,
        size_t layer_idx,
        size_t bucket,
        size_t tp_rank) const;

#ifdef ENABLE_ATEN
    AotPackageRunner &runner(
        PiecewiseInductorSegmentId segment_id,
        size_t layer_idx,
        size_t bucket);
#endif

private:
    InductorSegmentRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<SegmentKey, std::string, SegmentKeyHash> package_paths_;

friend AotPackageRunner &lookup_inductor_runner(
    InductorSegmentRegistry &registry,
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank);
};

#ifdef ENABLE_ATEN
AotPackageRunner &lookup_inductor_runner(
    InductorSegmentRegistry &registry,
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank);
#endif

} // namespace infinicore::op::inductor_segment_impl
