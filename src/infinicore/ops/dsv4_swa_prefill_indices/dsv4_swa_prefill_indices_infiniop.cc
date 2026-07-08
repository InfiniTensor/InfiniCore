#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_swa_prefill_indices.hpp"
#include "infiniop/ops/dsv4_swa_prefill_indices.h"
namespace infinicore::op::dsv4_swa_prefill_indices_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SwaPrefillIndices, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, indices;
};
void *plan(Tensor indices, int window_size) {
    size_t seed = hash_combine(indices, window_size);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, Dsv4SwaPrefillIndices, seed, indices->desc(), window_size);
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SwaPrefillIndices, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(indices)};
}
void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4SwaPrefillIndices(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->indices->data(), context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SwaPrefillIndices, &plan, &run, &cleanup);
} // namespace infinicore::op::dsv4_swa_prefill_indices_impl::infiniop
