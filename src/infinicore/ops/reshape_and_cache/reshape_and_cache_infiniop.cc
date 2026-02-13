#include "infinicore/ops/reshape_and_cache.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::reshape_and_cache_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, ReshapeAndCache, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor key;
    graph::GraphTensor value;
    graph::GraphTensor key_cache;
    graph::GraphTensor value_cache;
    graph::GraphTensor slot_mapping;
    graph::GraphTensor k_scale;
    graph::GraphTensor v_scale;
    std::string kv_cache_dtype;
};

void *plan(Tensor &key,
           Tensor &value,
           Tensor &key_cache,
           Tensor &value_cache,
           Tensor &slot_mapping,
           const std::string &kv_cache_dtype,
           Tensor &k_scale,
           Tensor &v_scale) {
    size_t seed = hash_combine(key, value, key_cache, value_cache, slot_mapping);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, ReshapeAndCache,
        seed,
        key->desc(), value->desc(), key_cache->desc(), value_cache->desc(),
        slot_mapping->desc(), kv_cache_dtype.c_str());

    INFINIOP_WORKSPACE_TENSOR(workspace, ReshapeAndCache, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(key),
        graph::GraphTensor(value),
        graph::GraphTensor(key_cache),
        graph::GraphTensor(value_cache),
        graph::GraphTensor(slot_mapping),
        graph::GraphTensor(k_scale),
        graph::GraphTensor(v_scale),
        kv_cache_dtype};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopReshapeAndCache(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->key->data(),
            p->value->data(),
            p->key_cache->data(),
            p->value_cache->data(),
            p->slot_mapping->data(),
            p->kv_cache_dtype.c_str(),
            p->k_scale->data(),
            p->v_scale->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(ReshapeAndCache, &plan, &run, &cleanup);

} // namespace infinicore::op::reshape_and_cache_impl::infiniop
