#include "infinicore/ops/moe_topk_sigmoid.hpp"

#include "../infiniop_impl.hpp"

#include <optional>

namespace infinicore::op::moe_topk_sigmoid_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, MoeTopkSigmoid, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor gating_output;
    std::optional<graph::GraphTensor> correction_bias;
};

void *plan(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &gating_output,
           const Tensor &correction_bias,
           const bool renormalize) {
    size_t seed = hash_combine(topk_weights, topk_indices, gating_output, correction_bias, renormalize);
    infiniopTensorDescriptor_t correction_bias_desc = correction_bias ? correction_bias->desc() : nullptr;
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        MoeTopkSigmoid,
        seed,
        topk_weights->desc(),
        topk_indices->desc(),
        gating_output->desc(),
        correction_bias_desc,
        renormalize);
    INFINIOP_WORKSPACE_TENSOR(workspace, MoeTopkSigmoid, descriptor);

    std::optional<graph::GraphTensor> correction_bias_graph;
    if (correction_bias) {
        correction_bias_graph.emplace(correction_bias);
    }
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(gating_output),
        correction_bias_graph};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopMoeTopkSigmoid(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->topk_weights->data(),
        planned->topk_indices->data(),
        planned->gating_output->data(),
        planned->correction_bias ? (*planned->correction_bias)->data() : nullptr,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MoeTopkSigmoid, &plan, &run, cleanup);

} // namespace infinicore::op::moe_topk_sigmoid_impl::infiniop
