#include "inductor_segment_registry.hpp"

#include <limits>
#include <stdexcept>

#ifdef ENABLE_ATEN
#include "aot_package_runner.hpp"
#endif

namespace infinicore::op::inductor_segment_impl {

namespace {

thread_local size_t g_lookup_tp_rank_override = std::numeric_limits<size_t>::max();
TpRankResolver g_tp_rank_resolver = nullptr;
ValidSeqLenResolver g_valid_seq_len_resolver = nullptr;

#ifdef ENABLE_ATEN
PreAttnWeightResolver g_pre_attn_weight_resolver;
#endif

size_t normalize_register_layer_idx(size_t layer_idx, bool layer_agnostic) {
    if (layer_agnostic) {
        return kLayerAgnosticIdx;
    }
    return layer_idx;
}

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

void set_piecewise_valid_seq_len_resolver(ValidSeqLenResolver resolver) {
    g_valid_seq_len_resolver = resolver;
}

size_t current_piecewise_valid_seq_len() {
    if (g_valid_seq_len_resolver != nullptr) {
        return g_valid_seq_len_resolver();
    }
    return 0;
}

#ifdef ENABLE_ATEN
void set_pre_attn_aten_weight_resolver(PreAttnWeightResolver resolver) {
    g_pre_attn_weight_resolver = std::move(resolver);
}

PreAttnExternalWeights resolve_pre_attn_weights(size_t layer_idx) {
    if (!g_pre_attn_weight_resolver) {
        throw std::runtime_error(
            "InductorSegment: pre_attn weight resolver not registered (layer-agnostic package)");
    }
    return g_pre_attn_weight_resolver(layer_idx);
}
#endif

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
    size_t tp_rank,
    bool layer_agnostic) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t reg_layer = normalize_register_layer_idx(layer_idx, layer_agnostic);
    SegmentKey key{segment_id, reg_layer, bucket, tp_rank};
    package_paths_[key] = package_path;
    layer_agnostic_flags_[key] = layer_agnostic;
#ifdef ENABLE_ATEN
    erase_inductor_runner(key);
#endif
}

void InductorSegmentRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    package_paths_.clear();
    layer_agnostic_flags_.clear();
#ifdef ENABLE_ATEN
    clear_inductor_runners();
#endif
}

bool InductorSegmentRegistry::has_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank,
    bool layer_agnostic) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t reg_layer = normalize_register_layer_idx(layer_idx, layer_agnostic);
    SegmentKey key{segment_id, reg_layer, bucket, tp_rank};
    return package_paths_.find(key) != package_paths_.end();
}

bool InductorSegmentRegistry::package_is_layer_agnostic(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank) const {
    std::lock_guard<std::mutex> lock(mutex_);
    SegmentKey exact{segment_id, layer_idx, bucket, tp_rank};
    auto it = layer_agnostic_flags_.find(exact);
    if (it != layer_agnostic_flags_.end() && it->second) {
        return true;
    }
    SegmentKey agnostic{segment_id, kLayerAgnosticIdx, bucket, tp_rank};
    it = layer_agnostic_flags_.find(agnostic);
    return it != layer_agnostic_flags_.end() && it->second;
}

#ifdef ENABLE_ATEN
namespace {

std::unordered_map<SegmentKey, std::unique_ptr<AotPackageRunner>, SegmentKeyHash> &
runner_cache() {
    static std::unordered_map<SegmentKey, std::unique_ptr<AotPackageRunner>, SegmentKeyHash> cache;
    return cache;
}

} // namespace

void erase_inductor_runner(const SegmentKey &key) {
    runner_cache().erase(key);
}

void clear_inductor_runners() {
    runner_cache().clear();
}

AotPackageRunner &InductorSegmentRegistry::runner(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    bool *layer_agnostic_out) {
    const size_t tp_rank = current_tensor_parallel_rank();
    return lookup_inductor_runner(
        *this, segment_id, layer_idx, bucket, tp_rank, layer_agnostic_out);
}

AotPackageRunner &lookup_inductor_runner(
    InductorSegmentRegistry &registry,
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank,
    bool *layer_agnostic_out) {
    bool layer_agnostic = false;
    std::string package_path;
    {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        SegmentKey exact{segment_id, layer_idx, bucket, tp_rank};
        auto path_it = registry.package_paths_.find(exact);
        if (path_it != registry.package_paths_.end()) {
            package_path = path_it->second;
            auto flag_it = registry.layer_agnostic_flags_.find(exact);
            layer_agnostic = flag_it != registry.layer_agnostic_flags_.end() && flag_it->second;
        } else {
            SegmentKey agnostic{segment_id, kLayerAgnosticIdx, bucket, tp_rank};
            path_it = registry.package_paths_.find(agnostic);
            if (path_it == registry.package_paths_.end()) {
                throw std::runtime_error(
                    "InductorSegment: no AOT package registered for segment="
                    + std::to_string(static_cast<int>(segment_id))
                    + " layer=" + std::to_string(layer_idx)
                    + " bucket=" + std::to_string(bucket)
                    + " tp_rank=" + std::to_string(tp_rank));
            }
            package_path = path_it->second;
            layer_agnostic = true;
        }
    }
    if (layer_agnostic_out != nullptr) {
        *layer_agnostic_out = layer_agnostic;
    }

    SegmentKey cache_key{
        segment_id,
        layer_agnostic ? kLayerAgnosticIdx : layer_idx,
        bucket,
        tp_rank,
    };
    auto &cache = runner_cache();
    auto runner_it = cache.find(cache_key);
    if (runner_it == cache.end()) {
        auto inserted = cache.emplace(
            cache_key,
            std::make_unique<AotPackageRunner>(package_path));
        runner_it = inserted.first;
    }
    return *runner_it->second;
}
#endif

void register_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    const std::string &package_path,
    size_t tp_rank,
    bool layer_agnostic) {
    InductorSegmentRegistry::instance().register_package(
        segment_id, layer_idx, bucket, package_path, tp_rank, layer_agnostic);
}

void clear_packages() {
    InductorSegmentRegistry::instance().clear();
}

bool has_package(
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket) {
    const size_t tp_rank = current_tensor_parallel_rank();
    auto &registry = InductorSegmentRegistry::instance();
    if (registry.has_package(segment_id, layer_idx, bucket, tp_rank, false)) {
        return true;
    }
    return registry.has_package(
        segment_id, kLayerAgnosticIdx, bucket, tp_rank, true);
}

} // namespace infinicore::op::inductor_segment_impl
