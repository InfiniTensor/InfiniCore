#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_per_token_quant_int8.hpp"
#include "infiniop/ops/dsv4_per_token_quant_int8.h"

namespace infinicore::op::dsv4_per_token_quant_int8_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4PerTokenQuantInt8, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, q, scale, x;
};

void *plan(Tensor q, Tensor scale, const Tensor &x) {
    size_t seed = hash_combine(q, scale, x);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4PerTokenQuantInt8,
        seed, q->desc(), scale->desc(), x->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4PerTokenQuantInt8, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(q), graph::GraphTensor(scale), graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4PerTokenQuantInt8(
        p->descriptor->desc,
        p->workspace->data(),
        p->workspace->numel(),
        p->q->data(),
        p->scale->data(),
        p->x->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4PerTokenQuantInt8, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_per_token_quant_int8_impl::infiniop
