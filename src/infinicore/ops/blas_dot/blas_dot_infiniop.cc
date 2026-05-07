#include "infinicore/ops/blas_dot.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::blas_dot_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, BlasDot, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, y, result;
};

void *plan(const Tensor &x, const Tensor &y, Tensor result) {
    size_t seed = hash_combine(x, y, result);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, BlasDot,
        seed,
        x->desc(), y->desc(), result->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, BlasDot, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(y),
        graph::GraphTensor(result)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopBlasDot(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x->data(),
        planned->y->data(),
        planned->result->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(BlasDot, &plan, &run, &cleanup);

} // namespace infinicore::op::blas_dot_impl::infiniop
