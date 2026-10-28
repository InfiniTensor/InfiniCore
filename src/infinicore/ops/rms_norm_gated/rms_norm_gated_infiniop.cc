#include "../infiniop_impl.hpp"
#include "infinicore/ops/rms_norm_gated.hpp"
#include "infiniop/ops/rms_norm_gated.h"
namespace infinicore::op::rms_norm_gated_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, RMSNormGated, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, x, gate, weight;
};
void *plan(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &weight, float epsilon) {
    size_t key = hash_combine(out, x, gate, weight, epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, RMSNormGated, key, out->desc(), x->desc(), gate->desc(), weight->desc(), epsilon);
    INFINIOP_WORKSPACE_TENSOR(workspace, RMSNormGated, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(out), graph::GraphTensor(x), graph::GraphTensor(gate), graph::GraphTensor(weight)};
}
void run(void *meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(meta);
    INFINICORE_CHECK_ERROR(infiniopRMSNormGated(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->out->data(), p->x->data(), p->gate->data(), p->weight->data(), context::getStream()));
}
void cleanup(void **meta) {
    delete *reinterpret_cast<PlannedMeta **>(meta);
    *meta = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(RMSNormGated, &plan, &run, &cleanup);
} // namespace infinicore::op::rms_norm_gated_impl::infiniop
