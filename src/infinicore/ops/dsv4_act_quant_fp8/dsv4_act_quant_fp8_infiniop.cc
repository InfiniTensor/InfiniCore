#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_act_quant_fp8.hpp"
#include "infiniop/ops/dsv4_act_quant_fp8.h"
namespace infinicore::op::dsv4_act_quant_fp8_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4ActQuantFp8, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, xq, scale, x;
};
void *plan(Tensor xq, Tensor scale, const Tensor &x, float fp8_max) {
    size_t seed = hash_combine(xq, scale, x, fp8_max);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, Dsv4ActQuantFp8, seed, xq->desc(), scale->desc(), x->desc(), fp8_max);
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4ActQuantFp8, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(xq), graph::GraphTensor(scale), graph::GraphTensor(x)};
}
void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4ActQuantFp8(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->xq->data(), p->scale->data(), p->x->data(), context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4ActQuantFp8, &plan, &run, &cleanup);
} // namespace infinicore::op::dsv4_act_quant_fp8_impl::infiniop
