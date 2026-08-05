#include "infinicore/ops/tanh.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::tanh_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Tanh, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input;
};

void *plan(Tensor output, const Tensor &input) {
    size_t seed = hash_combine(output, input);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Tanh,
        seed,
        output->desc(),
        input->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Tanh, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopTanh(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Tanh, &plan, &run, &cleanup);

} // namespace infinicore::op::tanh_impl::infiniop
