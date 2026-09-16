#include "infinicore/ops/w4a8_moe_shuffle.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::w4a8_moe_shuffle_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, W4A8MoeShuffle, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor output;
    graph::GraphTensor input;
};

void *plan(Tensor output, const Tensor &input) {
    const size_t seed = hash_combine(output, input);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, W4A8MoeShuffle, seed,
        output->desc(), input->desc());
    return new PlannedMeta{descriptor, graph::GraphTensor(output),
                           graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopW4A8MoeShuffle(
        planned->descriptor->desc, planned->output->data(),
        planned->input->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(
    W4A8MoeShuffle, &plan, &run, &cleanup);

} // namespace infinicore::op::w4a8_moe_shuffle_impl::infiniop
