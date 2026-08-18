#include "infinicore/ops/paged_caching.hpp"

#include "../../infiniop_impl.hpp"

#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <c10/hip/HIPGuard.h>

#include <optional>
#include <string>
#endif

namespace infinicore::op::paged_caching_impl::lightop_hygon {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PagedCaching, 100);

struct PlannedMeta {
    bool use_hygon_lightop;
    std::shared_ptr<Descriptor> descriptor;
    std::optional<graph::GraphTensor> workspace;
    graph::GraphTensor k_cache, v_cache, k, v, slot_mapping;
#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
    at::Tensor k_scale, v_scale;
#endif
};

#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
bool is_hygon_vllm_cache_layout(const Tensor &k_cache, const Tensor &v_cache) {
    if (k_cache->device().getType() != Device::Type::HYGON) {
        return false;
    }
    const auto &k_shape = k_cache->shape();
    const auto &v_shape = v_cache->shape();
    return k_shape.size() == 4 && v_shape.size() == 4
        && k_shape[0] == v_shape[0]
        && k_shape[1] == v_shape[1]
        && k_shape[2] == v_shape[3]
        && k_shape[3] == v_shape[2]
        && k_shape[2] == 64;
}
#endif

void *plan(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
    if (is_hygon_vllm_cache_layout(k_cache, v_cache)) {
        infinicore::adaptor::lightop::preload_reshape_and_cache_cuda();
        c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
        auto options = infinicore::adaptor::to_aten_tensor(k).options().dtype(at::kFloat);
        return new PlannedMeta{
            true,
            nullptr,
            std::nullopt,
            graph::GraphTensor(k_cache),
            graph::GraphTensor(v_cache),
            graph::GraphTensor(k),
            graph::GraphTensor(v),
            graph::GraphTensor(slot_mapping),
            at::ones({1}, options),
            at::ones({1}, options)};
    }
#endif

    size_t key = hash_combine(k_cache, v_cache, k, v, slot_mapping);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, PagedCaching,
        key, k_cache->desc(), v_cache->desc(), k->desc(), v->desc(), slot_mapping->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, PagedCaching, descriptor);

    return new PlannedMeta {
        false,
            descriptor,
            graph::GraphTensor(workspace),
            graph::GraphTensor(k_cache),
            graph::GraphTensor(v_cache),
            graph::GraphTensor(k),
            graph::GraphTensor(v),
            graph::GraphTensor(slot_mapping)
#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
                ,
            at::Tensor{}, at::Tensor {
        }
#endif
    };
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
    if (p->use_hygon_lightop) {
        c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
        auto k = infinicore::adaptor::to_aten_tensor(p->k);
        auto v = infinicore::adaptor::to_aten_tensor(p->v);
        auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
        auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
        auto slot_mapping = infinicore::adaptor::to_aten_tensor(p->slot_mapping);
        static const std::string kv_cache_dtype = "auto";
        infinicore::adaptor::lightop::reshape_and_cache_cuda(
            k,
            v,
            k_cache,
            v_cache,
            slot_mapping,
            kv_cache_dtype,
            p->k_scale,
            p->v_scale);
        return;
    }
#endif

    auto &workspace = p->workspace.value();
    INFINICORE_CHECK_ERROR(
        infiniopPagedCaching(
            p->descriptor->desc,
            workspace->data(),
            workspace->numel(),
            p->k_cache->data(),
            p->v_cache->data(),
            p->k->data(),
            p->v->data(),
            p->slot_mapping->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    PagedCaching::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    PagedCaching::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    PagedCaching::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::paged_caching_impl::lightop_hygon
