#include "infinicore/ops/dsv4_sglang_hash_topk.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_hash_topk.h"

namespace infinicore::op::dsv4_sglang_hash_topk_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangHashTopk, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, router_logits, input_ids, tid2eid, topk_weights, topk_ids;
};

void *plan(const Tensor &router_logits, const Tensor &input_ids, const Tensor &tid2eid, Tensor topk_weights, Tensor topk_ids, float routed_scaling_factor) {
    size_t seed = hash_combine(router_logits, input_ids, tid2eid, topk_weights, topk_ids, routed_scaling_factor);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangHashTopk,
        seed,
        router_logits->desc(), input_ids->desc(), tid2eid->desc(), topk_weights->desc(), topk_ids->desc(), routed_scaling_factor);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangHashTopk, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(router_logits),
        graph::GraphTensor(input_ids),
        graph::GraphTensor(tid2eid),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_ids)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangHashTopk(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->router_logits->data(),
        planned->input_ids->data(),
        planned->tid2eid->data(),
        planned->topk_weights->data(),
        planned->topk_ids->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangHashTopk, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_hash_topk_impl::infiniop
