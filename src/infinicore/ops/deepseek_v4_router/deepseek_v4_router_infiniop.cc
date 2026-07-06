#include "infinicore/ops/deepseek_v4_router.hpp"

#include "../infiniop_impl.hpp"

#include <optional>

namespace infinicore::op::deepseek_v4_topk_router_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4TopkRouter, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor logits;
    std::optional<graph::GraphTensor> bias;
};

void *plan(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &logits,
           const Tensor &bias,
           bool renormalize) {
    size_t seed = hash_combine(topk_weights, topk_indices, logits, bias, renormalize);
    infiniopTensorDescriptor_t bias_desc = bias ? bias->desc() : nullptr;
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4TopkRouter,
        seed,
        topk_weights->desc(),
        topk_indices->desc(),
        logits->desc(),
        bias_desc,
        renormalize);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4TopkRouter, descriptor);

    std::optional<graph::GraphTensor> bias_graph;
    if (bias) {
        bias_graph.emplace(bias);
    }
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(logits),
        bias_graph};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4TopkRouter(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->topk_weights->data(),
        planned->topk_indices->data(),
        planned->logits->data(),
        planned->bias ? (*planned->bias)->data() : nullptr,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4TopkRouter, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_topk_router_impl::infiniop

namespace infinicore::op::deepseek_v4_hash_router_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4HashRouter, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor logits;
    graph::GraphTensor input_ids;
    graph::GraphTensor tid2eid;
};

void *plan(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &logits,
           const Tensor &input_ids,
           const Tensor &tid2eid,
           bool renormalize) {
    size_t seed = hash_combine(topk_weights, topk_indices, logits, input_ids, tid2eid, renormalize);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4HashRouter,
        seed,
        topk_weights->desc(),
        topk_indices->desc(),
        logits->desc(),
        input_ids->desc(),
        tid2eid->desc(),
        renormalize);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4HashRouter, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(logits),
        graph::GraphTensor(input_ids),
        graph::GraphTensor(tid2eid)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4HashRouter(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->topk_weights->data(),
        planned->topk_indices->data(),
        planned->logits->data(),
        planned->input_ids->data(),
        planned->tid2eid->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4HashRouter, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_hash_router_impl::infiniop
