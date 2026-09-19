#include "infinicore/ops/lightning_attention.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::lightning_attention_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, LightningAttention, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, initial_state, q, k, v, slope;
    graph::GraphTensor initial_state_indices;
    graph::GraphTensor final_state_indices;
};

void *plan(Tensor out,
           Tensor initial_state,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &slope,
           const Tensor &initial_state_indices,
           const Tensor &final_state_indices) {
    size_t seed = hash_combine(out,
                               initial_state,
                               q,
                               k,
                               v,
                               slope,
                               initial_state_indices,
                               final_state_indices);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, LightningAttention,
        seed,
        out->desc(),
        initial_state->desc(),
        q->desc(),
        k->desc(),
        v->desc(),
        slope->desc(),
        initial_state_indices->desc(),
        final_state_indices->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, LightningAttention, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(initial_state),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(slope),
        graph::GraphTensor(initial_state_indices),
        graph::GraphTensor(final_state_indices)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopLightningAttention(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->initial_state->data(),
        planned->q->data(),
        planned->k->data(),
        planned->v->data(),
        planned->slope->data(),
        planned->initial_state_indices->data(),
        planned->final_state_indices->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(LightningAttention, &plan, &run, &cleanup);

} // namespace infinicore::op::lightning_attention_impl::infiniop
