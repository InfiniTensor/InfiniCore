#include "infinicore/ops/paged_caching.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::paged_caching_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PagedCaching, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;

    graph::GraphTensor workspace, k_cache, v_cache, k, v, slot_mapping;
    std::optional<graph::GraphTensor> k_scale, v_scale;
};

void *plan(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping,
           std::optional<Tensor> k_scale, std::optional<Tensor> v_scale) {
    size_t key = hash_combine(k_cache, v_cache, k, v, slot_mapping, k_scale, v_scale);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, PagedCaching,
        key, k_cache->desc(), v_cache->desc(), k->desc(), v->desc(), slot_mapping->desc(),
        k_scale ? k_scale.value()->desc() : nullptr,
        v_scale ? v_scale.value()->desc() : nullptr);

    INFINIOP_WORKSPACE_TENSOR(workspace, PagedCaching, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(slot_mapping),
        k_scale ? std::optional<graph::GraphTensor>(graph::GraphTensor(*k_scale)) : std::nullopt,
        v_scale ? std::optional<graph::GraphTensor>(graph::GraphTensor(*v_scale)) : std::nullopt};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopPagedCaching(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->k_cache->data(),
            p->v_cache->data(),
            p->k->data(),
            p->v->data(),
            p->slot_mapping->data(),
            p->k_scale.has_value() ? p->k_scale.value()->data() : nullptr,
            p->v_scale.has_value() ? p->v_scale.value()->data() : nullptr,
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(PagedCaching, &plan, &run, &cleanup);

} // namespace infinicore::op::paged_caching_impl::infiniop
