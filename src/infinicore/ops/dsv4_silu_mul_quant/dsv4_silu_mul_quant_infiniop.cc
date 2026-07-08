#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_silu_mul_quant.hpp"
#include "infiniop/ops/dsv4_silu_mul_quant.h"
namespace infinicore::op::dsv4_silu_mul_quant_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SiluMulQuant, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, q, scale, gate, up;
};
void *plan(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up) {
    size_t seed = hash_combine(q, scale, gate, up);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, Dsv4SiluMulQuant, seed, q->desc(), scale->desc(), gate->desc(), up->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SiluMulQuant, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(q), graph::GraphTensor(scale), graph::GraphTensor(gate), graph::GraphTensor(up)};
}
void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4SiluMulQuant(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->q->data(), p->scale->data(), p->gate->data(), p->up->data(), context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SiluMulQuant, &plan, &run, &cleanup);
} // namespace infinicore::op::dsv4_silu_mul_quant_impl::infiniop
