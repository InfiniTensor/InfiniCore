#include "infinicore/ops/ones.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::ones_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Ones, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output;
};

void *plan(Tensor output) {
    size_t seed = hash_combine(output);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Ones,
        seed,
        output->desc(), output->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Ones, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopOnes(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->output->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Ones, &plan, &run, &cleanup);

} // namespace infinicore::op::ones_impl::infiniop
