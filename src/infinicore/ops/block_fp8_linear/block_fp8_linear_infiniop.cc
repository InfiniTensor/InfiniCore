#include "infinicore/ops/block_fp8_linear.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::block_fp8_linear_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, BlockFP8Linear, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    graph::GraphTensor weight_scale;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &weight,
           const Tensor &weight_scale) {
    size_t seed = hash_combine(output, input, weight, weight_scale);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        BlockFP8Linear,
        seed,
        output->desc(),
        input->desc(),
        weight->desc(),
        weight_scale->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, BlockFP8Linear, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        graph::GraphTensor(weight),
        graph::GraphTensor(weight_scale)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopBlockFP8Linear(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        planned->weight->data(),
        planned->weight_scale->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(BlockFP8Linear, &plan, &run, &cleanup);

} // namespace infinicore::op::block_fp8_linear_impl::infiniop
