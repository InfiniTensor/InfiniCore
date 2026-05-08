#include "infinicore/ops/swap.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::swap_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Swap, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, y;
};

void *plan(Tensor x, Tensor y) {
    size_t seed = hash_combine(x, y);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Swap,
        seed,
        x->desc(), y->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Swap, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(y)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSwap(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x->data(),
        planned->y->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Swap, &plan, &run, &cleanup);

} // namespace infinicore::op::swap_impl::infiniop
