#include "infinicore/ops/rot.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::rot_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Rot, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, y, c, s;
};

void *plan(Tensor x, Tensor y, const Tensor &c, const Tensor &s) {
    size_t seed = hash_combine(x, y, c, s);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Rot,
        seed,
        x->desc(), y->desc(), c->desc(), s->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Rot, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(y),
        graph::GraphTensor(c),
        graph::GraphTensor(s)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopRot(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x->data(),
        planned->y->data(),
        planned->c->data(),
        planned->s->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Rot, &plan, &run, &cleanup);

} // namespace infinicore::op::rot_impl::infiniop
