#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_sglang_store_indexer.hpp"
#include "infiniop/ops/dsv4_sglang_store_indexer.h"
namespace infinicore::op::dsv4_sglang_store_indexer_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangStoreIndexer, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, input, cache, indices;
};
void *plan(const Tensor &input, Tensor cache, const Tensor &indices) {
    size_t seed = hash_combine(input, cache, indices);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangStoreIndexer,
        seed,
        input->desc(), cache->desc(), indices->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangStoreIndexer, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(input), graph::GraphTensor(cache), graph::GraphTensor(indices)};
}
void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4SglangStoreIndexer(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->input->data(), planned->cache->data(), planned->indices->data(),
        context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangStoreIndexer, &plan, &run, &cleanup);
} // namespace infinicore::op::dsv4_sglang_store_indexer_impl::infiniop
