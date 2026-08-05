#include "../infiniop_impl.hpp"
#include "infinicore/ops/select_last_token_hidden.hpp"

namespace infinicore::op::select_last_token_hidden_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, SelectLastTokenHidden, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor output, hidden_states, input_offsets;
};

void *plan(Tensor output, const Tensor &hidden_states, const Tensor &input_offsets) {
    const size_t seed = hash_combine(output, hidden_states, input_offsets);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        SelectLastTokenHidden,
        seed,
        output->desc(),
        hidden_states->desc(),
        input_offsets->desc());

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(output),
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(input_offsets)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopSelectLastTokenHidden(
        planned->descriptor->desc,
        planned->output->data(),
        planned->hidden_states->data(),
        planned->input_offsets->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    SelectLastTokenHidden,
    &plan,
    &run,
    &cleanup);

} // namespace infinicore::op::select_last_token_hidden_impl::infiniop
