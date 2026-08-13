#include "../infiniop_impl.hpp"
#include "infinicore/ops/linear_allreduce.hpp"

#include <array>
#include <string>

namespace infinicore::op::linear_allreduce_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, MatmulAllReduce, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    std::optional<graph::GraphTensor> bias;
};

void *plan(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    std::array<char, INFINICCL_COMM_NAME_MAX_LENGTH> group_name{};
    INFINICORE_CHECK_ERROR(infinicclGetCommName(
        communicator, group_name.data(), group_name.size()));

    size_t seed = hash_combine(
        output, input, weight, bias, std::string(group_name.data()));
    auto bias_desc = bias ? (*bias)->desc() : nullptr;

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, MatmulAllReduce,
        seed, output->desc(), input->desc(), weight->desc(),
        bias_desc, group_name.data());

    INFINIOP_WORKSPACE_TENSOR(
        workspace, MatmulAllReduce, descriptor);

    std::optional<graph::GraphTensor> graph_bias;
    if (bias) {
        graph_bias.emplace(*bias);
    }
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        graph::GraphTensor(weight),
        std::move(graph_bias)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    const void *bias = planned->bias
                         ? (*planned->bias)->data()
                         : nullptr;
    INFINICORE_CHECK_ERROR(infiniopMatmulAllReduce(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        planned->weight->data(),
        bias,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    LinearAllReduce::plan_dispatcher().registerDevice(
        Device::Type::ASCEND, &plan);
    LinearAllReduce::run_dispatcher().registerDevice(
        Device::Type::ASCEND, &run);
    LinearAllReduce::cleanup_dispatcher().registerDevice(
        Device::Type::ASCEND, &cleanup);
    return true;
}();

} // namespace infinicore::op::linear_allreduce_impl::infiniop
