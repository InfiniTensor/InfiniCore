#ifdef ENABLE_ATEN

#include "aot_package_runner.hpp"
#include "inductor_segment_registry.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"

#include <stdexcept>

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif

namespace infinicore::op::inductor_segment_impl {

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

AotPackageRunner &lookup_inductor_runner(
    InductorSegmentRegistry &registry,
    PiecewiseInductorSegmentId segment_id,
    size_t layer_idx,
    size_t bucket,
    size_t tp_rank) {
    std::lock_guard<std::mutex> lock(registry.mutex_);
    SegmentKey key{segment_id, layer_idx, bucket, tp_rank};
    auto path_it = registry.package_paths_.find(key);
    if (path_it == registry.package_paths_.end()) {
        throw std::runtime_error(
            "InductorSegment: no AOT package registered for segment="
            + std::to_string(static_cast<int>(segment_id))
            + " layer=" + std::to_string(layer_idx)
            + " bucket=" + std::to_string(bucket)
            + " tp_rank=" + std::to_string(tp_rank));
    }
    auto &cache = runner_cache();
    auto runner_it = cache.find(key);
    if (runner_it == cache.end()) {
        auto inserted = cache.emplace(
            key,
            std::make_unique<AotPackageRunner>(path_it->second));
        runner_it = inserted.first;
    }
    return *runner_it->second;
}

AotPackageRunner::AotPackageRunner(const std::string &package_path)
    : package_path_(package_path),
      loader_(std::make_unique<torch::inductor::AOTIModelPackageLoader>(
          package_path,
          "model",
          /*run_single_threaded=*/true)) {}

std::vector<at::Tensor> AotPackageRunner::run(
    const std::vector<at::Tensor> &inputs,
    void *stream_handle) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    void *stream = stream_handle != nullptr
                       ? stream_handle
                       : infinicore::context::getStream();
    return loader_->run(inputs, stream);
}

void AotPackageRunner::warmup(const std::vector<at::Tensor> &inputs) {
    (void)run(inputs, nullptr);
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::getCurrentCUDAStream().synchronize();
#endif
}

} // namespace infinicore::op::inductor_segment_impl

#endif // ENABLE_ATEN
