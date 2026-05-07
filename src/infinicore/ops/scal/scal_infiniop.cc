#include "infinicore/ops/scal.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::scal_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Scal, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, x;
};

void *plan(const Tensor &alpha, Tensor x) {
    size_t seed = hash_combine(alpha, x);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Scal,
        seed,
        alpha->desc(), x->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Scal, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopScal(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->x->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Scal, &plan, &run, &cleanup);

} // namespace infinicore::op::scal_impl::infiniop
