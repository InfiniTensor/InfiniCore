#include "infinicore/ops/topksoftmax.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::topksoftmax_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Topksoftmax, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, values, indices, x;
    size_t topk;
    int norm;
};

void *plan(Tensor values, Tensor indices, const Tensor &x, const size_t topk, const int norm) {
    size_t seed = hash_combine(values, indices, x);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Topksoftmax, seed, x->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Topksoftmax, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(values),
        graph::GraphTensor(indices),
        graph::GraphTensor(x),
        topk,
        norm};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopTopksoftmax(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->values->data(),
        planned->indices->data(),
        planned->x->data(),
        planned->topk,
        planned->norm,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Topksoftmax, &plan, &run, cleanup);

} // namespace infinicore::op::topksoftmax_impl::infiniop
