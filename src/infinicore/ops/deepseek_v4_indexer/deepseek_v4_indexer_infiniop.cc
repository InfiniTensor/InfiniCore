#include "infinicore/ops/deepseek_v4_indexer.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::deepseek_v4_indexer_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4Indexer, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor indices;
    graph::GraphTensor q;
    graph::GraphTensor weights;
    graph::GraphTensor compressed;
    graph::GraphTensor positions;
};

void *plan(Tensor indices,
           const Tensor &q,
           const Tensor &weights,
           const Tensor &compressed,
           const Tensor &positions,
           size_t query_start,
           size_t compress_ratio) {
    size_t seed = hash_combine(indices, q, weights, compressed, positions, query_start, compress_ratio);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4Indexer,
        seed,
        indices->desc(),
        q->desc(),
        weights->desc(),
        compressed->desc(),
        positions->desc(),
        query_start,
        compress_ratio);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4Indexer, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(indices),
        graph::GraphTensor(q),
        graph::GraphTensor(weights),
        graph::GraphTensor(compressed),
        graph::GraphTensor(positions)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4Indexer(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->indices->data(),
        planned->q->data(),
        planned->weights->data(),
        planned->compressed->data(),
        planned->positions->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4Indexer, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_indexer_impl::infiniop
