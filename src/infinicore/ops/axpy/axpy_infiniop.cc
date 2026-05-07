#include "infinicore/ops/axpy.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::axpy_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Axpy, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, x, y;
};

void *plan(const Tensor &alpha, const Tensor &x, Tensor y) {
    size_t seed = hash_combine(y, alpha, x);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Axpy,
        seed,
        alpha->desc(), x->desc(), y->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Axpy, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(x),
        graph::GraphTensor(y)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopAxpy(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->x->data(),
        planned->y->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Axpy, &plan, &run, &cleanup);

} // namespace infinicore::op::axpy_impl::infiniop
