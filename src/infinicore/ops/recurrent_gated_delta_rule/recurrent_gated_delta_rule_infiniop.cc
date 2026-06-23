#include "infinicore/ops/recurrent_gated_delta_rule.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::recurrent_gated_delta_rule_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, RecurrentGatedDeltaRule, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, initial_state, q, k, v, g, beta;
    std::optional<graph::GraphTensor> final_state;
    std::optional<graph::GraphTensor> initial_state_indices;
    std::optional<graph::GraphTensor> final_state_indices;
};

void *plan(Tensor out,
           Tensor initial_state,
           std::optional<Tensor> final_state,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &g,
           const Tensor &beta,
           std::optional<Tensor> initial_state_indices,
           std::optional<Tensor> final_state_indices,
           bool use_qk_l2norm) {
    size_t seed = hash_combine(out,
                               initial_state,
                               final_state,
                               q,
                               k,
                               v,
                               g,
                               beta,
                               initial_state_indices,
                               final_state_indices,
                               use_qk_l2norm);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, RecurrentGatedDeltaRule,
        seed,
        out->desc(),
        initial_state->desc(),
        final_state.has_value() ? final_state.value()->desc() : nullptr,
        q->desc(),
        k->desc(),
        v->desc(),
        g->desc(),
        beta->desc(),
        initial_state_indices.has_value() ? initial_state_indices.value()->desc() : nullptr,
        final_state_indices.has_value() ? final_state_indices.value()->desc() : nullptr,
        use_qk_l2norm);

    INFINIOP_WORKSPACE_TENSOR(workspace, RecurrentGatedDeltaRule, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(initial_state),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(g),
        graph::GraphTensor(beta),
        final_state.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(final_state.value())) : std::nullopt,
        initial_state_indices.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(initial_state_indices.value())) : std::nullopt,
        final_state_indices.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(final_state_indices.value())) : std::nullopt};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopRecurrentGatedDeltaRule(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->initial_state->data(),
        planned->final_state.has_value() ? planned->final_state.value()->data() : nullptr,
        planned->q->data(),
        planned->k->data(),
        planned->v->data(),
        planned->g->data(),
        planned->beta->data(),
        planned->initial_state_indices.has_value() ? planned->initial_state_indices.value()->data() : nullptr,
        planned->final_state_indices.has_value() ? planned->final_state_indices.value()->data() : nullptr,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(RecurrentGatedDeltaRule, &plan, &run, &cleanup);

} // namespace infinicore::op::recurrent_gated_delta_rule_impl::infiniop
