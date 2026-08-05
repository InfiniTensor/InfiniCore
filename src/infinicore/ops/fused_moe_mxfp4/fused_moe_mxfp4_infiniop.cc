#include "infinicore/ops/fused_moe_mxfp4.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::fused_moe_mxfp4_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, FusedMoeMxfp4, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor selected_experts;
    graph::GraphTensor routing_weights;
    graph::GraphTensor w13_packed;
    graph::GraphTensor w13_scale;
    graph::GraphTensor w2_packed;
    graph::GraphTensor w2_scale;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &selected_experts,
           const Tensor &routing_weights,
           const Tensor &w13_packed,
           const Tensor &w13_scale,
           const Tensor &w2_packed,
           const Tensor &w2_scale,
           FusedMoeActivation activation) {
    size_t seed = hash_combine(
        output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale,
        static_cast<int>(activation));
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        FusedMoeMxfp4,
        seed,
        output->desc(),
        input->desc(),
        selected_experts->desc(),
        routing_weights->desc(),
        w13_packed->desc(),
        w13_scale->desc(),
        w2_packed->desc(),
        w2_scale->desc(),
        static_cast<infiniopFusedMoeActivation_t>(activation));
    INFINIOP_WORKSPACE_TENSOR(workspace, FusedMoeMxfp4, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        graph::GraphTensor(selected_experts),
        graph::GraphTensor(routing_weights),
        graph::GraphTensor(w13_packed),
        graph::GraphTensor(w13_scale),
        graph::GraphTensor(w2_packed),
        graph::GraphTensor(w2_scale)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFusedMoeMxfp4(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        planned->selected_experts->data(),
        planned->routing_weights->data(),
        planned->w13_packed->data(),
        planned->w13_scale->data(),
        planned->w2_packed->data(),
        planned->w2_scale->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(FusedMoeMxfp4, &plan, &run, &cleanup);

} // namespace infinicore::op::fused_moe_mxfp4_impl::infiniop
