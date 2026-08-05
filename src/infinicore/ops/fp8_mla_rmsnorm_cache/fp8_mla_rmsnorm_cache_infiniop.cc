#include "../infiniop_impl.hpp"
#include "infinicore/ops/fp8_mla_rmsnorm_cache.hpp"

namespace infinicore::op::fp8_mla_rmsnorm_cache_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Fp8MlaRmsnormCache, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor cache, compressed_kv, norm_weight, rope, slot_mapping;
};

void *plan(
    Tensor cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps) {
    const size_t seed = hash_combine(
        cache, compressed_kv, norm_weight, rope, slot_mapping, eps);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Fp8MlaRmsnormCache, seed,
        cache->desc(), nullptr, compressed_kv->desc(),
        norm_weight->desc(), rope->desc(), slot_mapping->desc(), eps);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(cache),
        graph::GraphTensor(compressed_kv),
        graph::GraphTensor(norm_weight),
        graph::GraphTensor(rope),
        graph::GraphTensor(slot_mapping)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFp8MlaRmsnormCache(
        planned->descriptor->desc,
        planned->cache->data(),
        nullptr,
        planned->compressed_kv->data(),
        planned->norm_weight->data(),
        planned->rope->data(),
        planned->slot_mapping->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    Fp8MlaRmsnormCache, &plan, &run, &cleanup);

namespace dual {

struct DualPlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor cache, vendor_cache, compressed_kv, norm_weight, rope,
        slot_mapping;
};

void *plan_dual(
    Tensor cache,
    Tensor vendor_cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps) {
    const size_t seed = hash_combine(
        cache, vendor_cache, compressed_kv, norm_weight, rope,
        slot_mapping, eps);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Fp8MlaRmsnormCache, seed,
        cache->desc(), vendor_cache->desc(), compressed_kv->desc(),
        norm_weight->desc(), rope->desc(), slot_mapping->desc(), eps);
    return new DualPlannedMeta{
        descriptor,
        graph::GraphTensor(cache),
        graph::GraphTensor(vendor_cache),
        graph::GraphTensor(compressed_kv),
        graph::GraphTensor(norm_weight),
        graph::GraphTensor(rope),
        graph::GraphTensor(slot_mapping)};
}

void run_dual(void *planned_meta) {
    auto *planned = reinterpret_cast<DualPlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFp8MlaRmsnormCache(
        planned->descriptor->desc,
        planned->cache->data(),
        planned->vendor_cache->data(),
        planned->compressed_kv->data(),
        planned->norm_weight->data(),
        planned->rope->data(),
        planned->slot_mapping->data(),
        context::getStream()));
}

void cleanup_dual(void **planned_meta_ptr) {
    delete *reinterpret_cast<DualPlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    Fp8MlaRmsnormDualCache, &plan_dual, &run_dual, &cleanup_dual);

} // namespace dual

} // namespace infinicore::op::fp8_mla_rmsnorm_cache_impl::infiniop
