#include "../infiniop_impl.hpp"
#include "infinicore/ops/fp8_indexer_quant.hpp"

namespace infinicore::op::fused_fp8_indexer_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, FusedFp8Indexer, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor q_fp8, weights_fp32, k_cache;
    graph::GraphTensor q_raw, k_weights, norm_weight, norm_bias;
    graph::GraphTensor positions, cos_sin_cache, slot_mapping;
};

void *plan(
    Tensor q_fp8, Tensor weights_fp32, Tensor k_cache,
    const Tensor &q_raw, const Tensor &k_weights,
    const Tensor &norm_weight, const Tensor &norm_bias,
    const Tensor &positions, const Tensor &cos_sin_cache,
    const Tensor &slot_mapping, size_t rope_dim,
    double eps, double weights_scale) {
    const size_t seed = hash_combine(
        q_fp8, weights_fp32, k_cache, q_raw, k_weights,
        norm_weight, norm_bias, positions, cos_sin_cache, slot_mapping,
        rope_dim, eps, weights_scale);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, FusedFp8Indexer, seed,
        q_fp8->desc(), weights_fp32->desc(), k_cache->desc(),
        q_raw->desc(), k_weights->desc(), norm_weight->desc(),
        norm_bias->desc(), positions->desc(), cos_sin_cache->desc(),
        slot_mapping->desc(), static_cast<uint64_t>(rope_dim),
        eps, weights_scale);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(q_fp8),
        graph::GraphTensor(weights_fp32),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(q_raw),
        graph::GraphTensor(k_weights),
        graph::GraphTensor(norm_weight),
        graph::GraphTensor(norm_bias),
        graph::GraphTensor(positions),
        graph::GraphTensor(cos_sin_cache),
        graph::GraphTensor(slot_mapping)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFusedFp8Indexer(
        planned->descriptor->desc,
        planned->q_fp8->data(),
        planned->weights_fp32->data(),
        planned->k_cache->data(),
        planned->q_raw->data(),
        planned->k_weights->data(),
        planned->norm_weight->data(),
        planned->norm_bias->data(),
        planned->positions->data(),
        planned->cos_sin_cache->data(),
        planned->slot_mapping->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    FusedFp8Indexer, &plan, &run, &cleanup);

} // namespace infinicore::op::fused_fp8_indexer_impl::infiniop
