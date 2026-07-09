#include "infinicore/ops/dsv4_per_token_group_quant_int8.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_per_token_group_quant_int8.h"

namespace infinicore::op::dsv4_per_token_group_quant_int8_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4PerTokenGroupQuantInt8, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, q, scale, x;
};

void *plan(Tensor q, Tensor scale, const Tensor &x, int group_size) {
    size_t seed = hash_combine(q, scale, x, group_size);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4PerTokenGroupQuantInt8,
        seed,
        q->desc(), scale->desc(), x->desc(), group_size);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4PerTokenGroupQuantInt8, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(q),
        graph::GraphTensor(scale),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4PerTokenGroupQuantInt8(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->q->data(),
        planned->scale->data(),
        planned->x->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4PerTokenGroupQuantInt8, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_per_token_group_quant_int8_impl::infiniop
