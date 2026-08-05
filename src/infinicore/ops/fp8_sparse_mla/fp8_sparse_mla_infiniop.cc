#include "../infiniop_impl.hpp"
#include "infinicore/ops/fp8_sparse_mla.hpp"

namespace infinicore::op::fp8_sparse_mla_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Fp8SparseMla, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, query, kv_cache, indices, topk_lens;
    float scale;
};

void *plan(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale) {
    const size_t seed = hash_combine(
        output, query, kv_cache, indices, topk_lens, scale);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        Fp8SparseMla,
        seed,
        output->desc(),
        query->desc(),
        kv_cache->desc(),
        indices->desc(),
        topk_lens->desc(),
        scale);
    INFINIOP_WORKSPACE_TENSOR(workspace, Fp8SparseMla, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(query),
        graph::GraphTensor(kv_cache),
        graph::GraphTensor(indices),
        graph::GraphTensor(topk_lens),
        scale};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFp8SparseMla(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->query->data(),
        planned->kv_cache->data(),
        planned->indices->data(),
        planned->topk_lens->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    Fp8SparseMla,
    &plan,
    &run,
    &cleanup);

} // namespace infinicore::op::fp8_sparse_mla_impl::infiniop
