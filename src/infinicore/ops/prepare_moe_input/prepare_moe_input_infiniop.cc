#include "infinicore/ops/prepare_moe_input.hpp"

#include "../infiniop_impl.hpp"

#include <optional>

namespace infinicore::op::prepare_moe_input_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PrepareMoeInput, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor expert_offsets;
    std::optional<graph::GraphTensor> blockscale_offsets;
    graph::GraphTensor problem_sizes1;
    graph::GraphTensor problem_sizes2;
    graph::GraphTensor input_permutation;
    graph::GraphTensor output_permutation;
    graph::GraphTensor topk_ids;
};

void *plan(Tensor expert_offsets,
           Tensor blockscale_offsets,
           Tensor problem_sizes1,
           Tensor problem_sizes2,
           Tensor input_permutation,
           Tensor output_permutation,
           const Tensor &topk_ids,
           const size_t num_experts,
           const size_t n,
           const size_t k) {
    size_t seed = hash_combine(
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        topk_ids,
        num_experts,
        n,
        k);

    infiniopTensorDescriptor_t blockscale_desc = blockscale_offsets ? blockscale_offsets->desc() : nullptr;
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        PrepareMoeInput,
        seed,
        expert_offsets->desc(),
        blockscale_desc,
        problem_sizes1->desc(),
        problem_sizes2->desc(),
        input_permutation->desc(),
        output_permutation->desc(),
        topk_ids->desc(),
        num_experts,
        n,
        k);

    INFINIOP_WORKSPACE_TENSOR(workspace, PrepareMoeInput, descriptor);

    std::optional<graph::GraphTensor> blockscale_offsets_graph;
    if (blockscale_offsets) {
        blockscale_offsets_graph.emplace(blockscale_offsets);
    }

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(expert_offsets),
        blockscale_offsets_graph,
        graph::GraphTensor(problem_sizes1),
        graph::GraphTensor(problem_sizes2),
        graph::GraphTensor(input_permutation),
        graph::GraphTensor(output_permutation),
        graph::GraphTensor(topk_ids)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopPrepareMoeInput(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->expert_offsets->data(),
        planned->blockscale_offsets ? (*planned->blockscale_offsets)->data() : nullptr,
        planned->problem_sizes1->data(),
        planned->problem_sizes2->data(),
        planned->input_permutation->data(),
        planned->output_permutation->data(),
        planned->topk_ids->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(PrepareMoeInput, &plan, &run, cleanup);

} // namespace infinicore::op::prepare_moe_input_impl::infiniop
