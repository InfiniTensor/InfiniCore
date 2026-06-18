#include "infinicore/ops/moe_fused_dense.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::moe_fused_dense_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, MoeFusedDense, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor output;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w13;
    graph::GraphTensor w2;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_ids;
    graph::GraphTensor sorted_token_ids;
    graph::GraphTensor expert_ids;
    graph::GraphTensor num_tokens_post_padded;
};

void *plan(Tensor output,
           const Tensor &hidden_states,
           const Tensor &w13,
           const Tensor &w2,
           const Tensor &topk_weights,
           const Tensor &topk_ids,
           const Tensor &sorted_token_ids,
           const Tensor &expert_ids,
           const Tensor &num_tokens_post_padded) {
    size_t seed = hash_combine(
        output, hidden_states, w13, w2, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        MoeFusedDense,
        seed,
        output->desc(),
        hidden_states->desc(),
        w13->desc(),
        w2->desc(),
        topk_weights->desc(),
        topk_ids->desc(),
        sorted_token_ids->desc(),
        expert_ids->desc(),
        num_tokens_post_padded->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, MoeFusedDense, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(w13),
        graph::GraphTensor(w2),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_ids),
        graph::GraphTensor(sorted_token_ids),
        graph::GraphTensor(expert_ids),
        graph::GraphTensor(num_tokens_post_padded)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopMoeFusedDense(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->hidden_states->data(),
        planned->w13->data(),
        planned->w2->data(),
        planned->topk_weights->data(),
        planned->topk_ids->data(),
        planned->sorted_token_ids->data(),
        planned->expert_ids->data(),
        planned->num_tokens_post_padded->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MoeFusedDense, &plan, &run, cleanup);

} // namespace infinicore::op::moe_fused_dense_impl::infiniop
