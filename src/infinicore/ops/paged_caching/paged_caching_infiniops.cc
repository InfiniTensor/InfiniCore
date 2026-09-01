#include "infinicore/ops/paged_caching.hpp"

#include <string>

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/reshape_and_cache_flash.h"

namespace infinicore::op::paged_caching_impl::infiniop {
void *plan(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping,
           std::optional<Tensor> k_scale, std::optional<Tensor> v_scale);
void run(void *planned_meta);
void cleanup(void **planned_meta_ptr);
} // namespace infinicore::op::paged_caching_impl::infiniop

namespace infinicore::op::paged_caching_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    // Populated on the flash path only.
    std::optional<TensorMeta> k, v, slot_mapping, scale, k_cache, v_cache;
    std::optional<graph::GraphTensor> k_tensor, v_tensor, slot_mapping_tensor, scale_tensor, k_cache_tensor, v_cache_tensor;
    void *fallback_meta;
};
} // namespace

void *plan(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping,
           std::optional<Tensor> k_scale, std::optional<Tensor> v_scale) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(k_cache->device().getType()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cache, v_cache, k, v, slot_mapping);

    // FP8 KV caches need on-write quantization, which the flash path does not
    // implement; fall back to the InfiniOP implementation.
    if (k_scale.has_value() || v_scale.has_value()) {
        return new PlannedMeta{
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            paged_caching_impl::infiniop::plan(k_cache, v_cache, k, v, slot_mapping, k_scale, v_scale)};
    }

    // The "auto" cache path ignores scales, but the canonical API requires them.
    auto scale = Tensor::empty({1}, DataType::F32, k_cache->device());
    auto k_cache_view = k_cache->permute({0, 2, 1, 3});
    auto v_cache_view = v_cache->permute({0, 2, 1, 3});

    return new PlannedMeta{
        TensorMeta(k), TensorMeta(v), TensorMeta(slot_mapping), TensorMeta(scale), TensorMeta(k_cache_view), TensorMeta(v_cache_view),
        graph::GraphTensor(k), graph::GraphTensor(v), graph::GraphTensor(slot_mapping), graph::GraphTensor(scale), graph::GraphTensor(k_cache), graph::GraphTensor(v_cache),
        nullptr};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    if (planned->fallback_meta != nullptr) {
        paged_caching_impl::infiniop::run(planned->fallback_meta);
        return;
    }
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::ReshapeAndCacheFlash::Call(
        handle,
        config,
        planned->k->tensor(*planned->k_tensor),
        planned->v->tensor(*planned->v_tensor),
        planned->slot_mapping->tensor(*planned->slot_mapping_tensor),
        planned->scale->tensor(*planned->scale_tensor),
        planned->scale->tensor(*planned->scale_tensor),
        std::string{"auto"},
        planned->k_cache->tensor(*planned->k_cache_tensor),
        planned->v_cache->tensor(*planned->v_cache_tensor));
}

void cleanup(void **planned_meta_ptr) {
    auto planned = *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    if (planned->fallback_meta != nullptr) {
        paged_caching_impl::infiniop::cleanup(&planned->fallback_meta);
    }
    delete planned;
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(PagedCaching::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(PagedCaching::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(PagedCaching::cleanup_dispatcher(), &cleanup);
    return true;
}();
} // namespace infinicore::op::paged_caching_impl::infiniops
#endif
