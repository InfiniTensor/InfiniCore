#include "inductor_segment_registry.hpp"

#include <limits>
#include <stdexcept>

namespace infinicore::op::inductor_segment_impl {

namespace {

thread_local size_t g_lookup_tp_rank_override = std::numeric_limits<size_t>::max();
TpRankResolver g_tp_rank_resolver = nullptr;

} // namespace

void set_lookup_tp_rank_override(size_t tp_rank) {
    g_lookup_tp_rank_override = tp_rank;
}

void clear_lookup_tp_rank_override() {
    g_lookup_tp_rank_override = std::numeric_limits<size_t>::max();
}

void set_tensor_parallel_rank_resolver(TpRankResolver resolver) {
    g_tp_rank_resolver = resolver;
}

size_t current_tensor_parallel_rank() {
    if (g_lookup_tp_rank_override != std::numeric_limits<size_t>::max()) {
        return g_lookup_tp_rank_override;
    }
    if (g_tp_rank_resolver != nullptr) {
        return g_tp_rank_resolver();
    }
    return 0;
}

SegmentKey make_segment_key(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket) {
    return SegmentKey{
        segment_id,
        layer_idx,
        bucket,
        current_tensor_parallel_rank(),
    };
}

InductorSegmentRegistry &InductorSegmentRegistry::instance() {
    static InductorSegmentRegistry registry;
    return registry;
}

void InductorSegmentRegistry::register_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    const std::string &package_path,
    size_t tp_rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    SegmentKey key{segment_id, layer_idx, bucket, tp_rank};
    package_paths_[key] = package_path;
#ifdef ENABLE_ATEN
    erase_inductor_runner(key);
#endif
}

void InductorSegmentRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    package_paths_.clear();
#ifdef ENABLE_ATEN
    clear_inductor_runners();
#endif
}

bool InductorSegmentRegistry::has_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank) const {
    std::lock_guard<std::mutex> lock(mutex_);
    SegmentKey key{segment_id, layer_idx, bucket, tp_rank};
    return package_paths_.find(key) != package_paths_.end();
}

#ifdef ENABLE_ATEN
AotPackageRunner &InductorSegmentRegistry::runner(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket) {
    return lookup_inductor_runner(
        *this, segment_id, layer_idx, bucket, current_tensor_parallel_rank());
}
#endif

void register_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    const std::string &package_path,
    size_t tp_rank) {
    InductorSegmentRegistry::instance().register_package(
        segment_id, layer_idx, bucket, package_path, tp_rank);
}

void clear_packages() {
    InductorSegmentRegistry::instance().clear();
}

bool has_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket) {
    return InductorSegmentRegistry::instance().has_package(
        segment_id, layer_idx, bucket, current_tensor_parallel_rank());
}

} // namespace infinicore::op::inductor_segment_impl
