#include "infinicore/ops/moe_fused_gate.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::moe_fused_gate_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, MoeFusedGate, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor input;
    graph::GraphTensor bias;
};

void *plan(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &input,
           const Tensor &bias,
           const size_t num_expert_group,
           const size_t topk_group,
           const size_t num_fused_shared_experts,
           const float routed_scaling_factor,
           const bool apply_routed_scaling_factor_on_output) {
    size_t seed = hash_combine(
        topk_weights,
        topk_indices,
        input,
        bias,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        MoeFusedGate,
        seed,
        topk_weights->desc(),
        topk_indices->desc(),
        input->desc(),
        bias->desc(),
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
    INFINIOP_WORKSPACE_TENSOR(workspace, MoeFusedGate, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(input),
        graph::GraphTensor(bias)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopMoeFusedGate(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->topk_weights->data(),
        planned->topk_indices->data(),
        planned->input->data(),
        planned->bias->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MoeFusedGate, &plan, &run, cleanup);

} // namespace infinicore::op::moe_fused_gate_impl::infiniop
