#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_add_rmsnorm_quant.hpp"
#include "infiniop/ops/dsv4_add_rmsnorm_quant.h"
namespace infinicore::op::dsv4_add_rmsnorm_quant_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4AddRMSNormQuant, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, res, q, scale, x, weight;
};
void *plan(Tensor res, Tensor q, Tensor scale, const Tensor &x, const Tensor &weight, float epsilon) {
    size_t seed = hash_combine(res, q, scale, x, weight, epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, Dsv4AddRMSNormQuant, seed, res->desc(), q->desc(), scale->desc(), x->desc(), weight->desc(), epsilon);
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4AddRMSNormQuant, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(res), graph::GraphTensor(q), graph::GraphTensor(scale), graph::GraphTensor(x), graph::GraphTensor(weight)};
}
void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4AddRMSNormQuant(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->res->data(), p->q->data(), p->scale->data(), p->x->data(), p->weight->data(), context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4AddRMSNormQuant, &plan, &run, &cleanup);
} // namespace infinicore::op::dsv4_add_rmsnorm_quant_impl::infiniop
