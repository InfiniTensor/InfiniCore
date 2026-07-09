#include "infinicore/ops/mul_scalar.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/mul_scalar.h"

namespace infinicore::op::mul_scalar_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, MulScalar, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a;
    double alpha;
};

void *plan(Tensor c, const Tensor &a, double alpha) {
    size_t seed = hash_combine(c, a, alpha);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, MulScalar,
        seed, c->desc(), a->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, MulScalar, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        alpha};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopMulScalar(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->c->data(),
        planned->a->data(),
        planned->alpha,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MulScalar, &plan, &run, &cleanup);

} // namespace infinicore::op::mul_scalar_impl::infiniop
