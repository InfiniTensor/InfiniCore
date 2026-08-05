#include "infinicore/ops/linear_mxfp4.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::linear_mxfp4_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, LinearMxfp4, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor packed_weight;
    graph::GraphTensor weight_scale;
    std::optional<graph::GraphTensor> bias;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &packed_weight,
           const Tensor &weight_scale,
           std::optional<Tensor> bias,
           float alpha) {
    size_t seed = hash_combine(output, input, packed_weight, weight_scale, bias, alpha);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        LinearMxfp4,
        seed,
        output->desc(),
        input->desc(),
        packed_weight->desc(),
        weight_scale->desc(),
        bias.has_value() ? bias.value()->desc() : nullptr,
        alpha);
    INFINIOP_WORKSPACE_TENSOR(workspace, LinearMxfp4, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        graph::GraphTensor(packed_weight),
        graph::GraphTensor(weight_scale),
        bias.has_value()
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(bias.value()))
            : std::nullopt};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopLinearMxfp4(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        planned->packed_weight->data(),
        planned->weight_scale->data(),
        planned->bias.has_value() ? planned->bias.value()->data() : nullptr,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(LinearMxfp4, &plan, &run, &cleanup);

} // namespace infinicore::op::linear_mxfp4_impl::infiniop
