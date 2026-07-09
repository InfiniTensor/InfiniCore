#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_sglang_main_k_norm_rope_flashmla.hpp"
#include "infiniop/ops/dsv4_sglang_main_k_norm_rope_flashmla.h"
namespace infinicore::op::dsv4_sglang_main_k_norm_rope_flashmla_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangMainKNormRopeFlashmla, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, kv, weight, freqs, positions, out_loc, cache;
};
void *plan(Tensor kv, const Tensor &weight, const Tensor &freqs, const Tensor &positions, const Tensor &out_loc, Tensor cache, double eps) {
    size_t seed = hash_combine(kv, weight, freqs, positions, out_loc, cache, eps);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangMainKNormRopeFlashmla,
        seed,
        kv->desc(), weight->desc(), freqs->desc(), positions->desc(), out_loc->desc(), cache->desc(), eps);
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangMainKNormRopeFlashmla, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(kv), graph::GraphTensor(weight), graph::GraphTensor(freqs), graph::GraphTensor(positions), graph::GraphTensor(out_loc), graph::GraphTensor(cache)};
}
void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4SglangMainKNormRopeFlashmla(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->kv->data(), planned->weight->data(), planned->freqs->data(), planned->positions->data(), planned->out_loc->data(), planned->cache->data(),
        context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangMainKNormRopeFlashmla, &plan, &run, &cleanup);
} // namespace infinicore::op::dsv4_sglang_main_k_norm_rope_flashmla_impl::infiniop
