#include "infinicore/ops/fused_moe.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::fused_moe_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, FusedMoe, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, input, token_selected_experts, token_final_scales, w1, w2;
    std::optional<graph::GraphTensor> b1;
    std::optional<graph::GraphTensor> b2;
};

void *plan(Tensor out,
           const Tensor &input,
           const Tensor &token_selected_experts,
           const Tensor &token_final_scales,
           const Tensor &w1,
           const Tensor &w2,
           std::optional<Tensor> b1,
           std::optional<Tensor> b2,
           FusedMoeActivation activation) {
    size_t seed = hash_combine(out, input, token_selected_experts, token_final_scales, w1, w2, b1, b2, static_cast<int>(activation));

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, FusedMoe,
        seed,
        out->desc(),
        input->desc(),
        token_selected_experts->desc(),
        token_final_scales->desc(),
        w1->desc(),
        w2->desc(),
        b1.has_value() ? b1.value()->desc() : nullptr,
        b2.has_value() ? b2.value()->desc() : nullptr,
        static_cast<infiniopFusedMoeActivation_t>(activation));

    INFINIOP_WORKSPACE_TENSOR(workspace, FusedMoe, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(token_selected_experts),
        graph::GraphTensor(token_final_scales),
        graph::GraphTensor(w1),
        graph::GraphTensor(w2),
        b1.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(b1.value())) : std::nullopt,
        b2.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(b2.value())) : std::nullopt};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopFusedMoe(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->input->data(),
        planned->token_selected_experts->data(),
        planned->token_final_scales->data(),
        planned->w1->data(),
        planned->w2->data(),
        planned->b1.has_value() ? planned->b1.value()->data() : nullptr,
        planned->b2.has_value() ? planned->b2.value()->data() : nullptr,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(FusedMoe, &plan, &run, &cleanup);

} // namespace infinicore::op::fused_moe_impl::infiniop
