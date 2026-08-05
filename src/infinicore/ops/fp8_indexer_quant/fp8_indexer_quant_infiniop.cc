#include "../infiniop_impl.hpp"
#include "infinicore/ops/fp8_indexer_quant.hpp"

namespace infinicore::op::fp8_indexer_quant_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Fp8IndexerQuant, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor q_fp8, weights_fp32, q, weights;
};

void *plan(
    Tensor q_fp8,
    Tensor weights_fp32,
    const Tensor &q,
    const Tensor &weights) {
    const size_t seed = hash_combine(q_fp8, weights_fp32, q, weights);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        Fp8IndexerQuant,
        seed,
        q_fp8->desc(),
        weights_fp32->desc(),
        q->desc(),
        weights->desc());
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(q_fp8),
        graph::GraphTensor(weights_fp32),
        graph::GraphTensor(q),
        graph::GraphTensor(weights)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFp8IndexerQuant(
        planned->descriptor->desc,
        planned->q_fp8->data(),
        planned->weights_fp32->data(),
        planned->q->data(),
        planned->weights->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    Fp8IndexerQuant,
    &plan,
    &run,
    &cleanup);

} // namespace infinicore::op::fp8_indexer_quant_impl::infiniop
