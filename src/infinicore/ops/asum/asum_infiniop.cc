#include "infinicore/ops/asum.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::asum_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Asum, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, result;
};

void *plan(const Tensor &x, Tensor result) {
    size_t seed = hash_combine(x, result);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Asum,
        seed,
        x->desc(), result->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Asum, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(result)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopAsum(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x->data(),
        planned->result->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Asum, &plan, &run, &cleanup);

} // namespace infinicore::op::asum_impl::infiniop
