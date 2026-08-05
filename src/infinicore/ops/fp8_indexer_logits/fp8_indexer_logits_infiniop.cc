#include "../infiniop_impl.hpp"
#include "infinicore/ops/fp8_indexer_logits.hpp"

namespace infinicore::op::fp8_indexer_logits_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Fp8IndexerLogits, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor logits, q_fp8, kv_cache, block_tables, weights_fp32;
    graph::GraphTensor positions, request_ids;
};

void *plan(
    Tensor logits,
    const Tensor &q_fp8,
    const Tensor &kv_cache,
    const Tensor &block_tables,
    const Tensor &weights_fp32,
    const Tensor &positions,
    const Tensor &request_ids) {
    const size_t seed = hash_combine(
        logits, q_fp8, kv_cache, block_tables, weights_fp32, positions, request_ids);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        Fp8IndexerLogits,
        seed,
        logits->desc(),
        q_fp8->desc(),
        kv_cache->desc(),
        block_tables->desc(),
        weights_fp32->desc(),
        positions->desc(),
        request_ids->desc());
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(logits),
        graph::GraphTensor(q_fp8),
        graph::GraphTensor(kv_cache),
        graph::GraphTensor(block_tables),
        graph::GraphTensor(weights_fp32),
        graph::GraphTensor(positions),
        graph::GraphTensor(request_ids)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFp8IndexerLogits(
        planned->descriptor->desc,
        planned->logits->data(),
        planned->q_fp8->data(),
        planned->kv_cache->data(),
        planned->block_tables->data(),
        planned->weights_fp32->data(),
        planned->positions->data(),
        planned->request_ids->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    Fp8IndexerLogits,
    &plan,
    &run,
    &cleanup);

} // namespace infinicore::op::fp8_indexer_logits_impl::infiniop
