#include "../infiniop_impl.hpp"
#include "infinicore/ops/broadcast_to.hpp"

namespace infinicore::op::broadcast_to_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, BroadcastTo, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x;
};

void *plan(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, BroadcastTo,
        seed,
        y->desc(), x->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, BroadcastTo, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopBroadcastTo(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->y->data(),
        planned->x->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(BroadcastTo, &plan, &run, &cleanup);

} // namespace infinicore::op::broadcast_to_impl::infiniop
