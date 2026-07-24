#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_silu_and_mul.hpp"
#include "infiniop/ops/dsv4_silu_and_mul.h"

namespace infinicore::op::dsv4_silu_and_mul_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SiluAndMul, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, gate, up;
};

void *plan(Tensor y, const Tensor &gate, const Tensor &up) {
    size_t seed = hash_combine(y, gate, up);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, Dsv4SiluAndMul, seed, y->desc(), gate->desc(), up->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SiluAndMul, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(y), graph::GraphTensor(gate), graph::GraphTensor(up)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4SiluAndMul(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->y->data(), p->gate->data(), p->up->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SiluAndMul, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_silu_and_mul_impl::infiniop
