#include "infinicore/ops/moe_fused_dense_ascend.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeFusedDenseAscend);

MoeFusedDenseAscend::MoeFusedDenseAscend(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    size_t global_num_experts,
    size_t local_expert_start,
    size_t local_num_experts) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, hidden_states, w13, w2, topk_weights, topk_ids);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, hidden_states, w13, w2,
        topk_weights, topk_ids, global_num_experts, local_expert_start,
        local_num_experts);
}

void MoeFusedDenseAscend::execute(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    size_t global_num_experts,
    size_t local_expert_start,
    size_t local_num_experts) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeFusedDenseAscend, output, hidden_states, w13, w2, topk_weights,
        topk_ids, global_num_experts, local_expert_start, local_num_experts);
}

Tensor moe_fused_dense_ascend(
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    size_t global_num_experts,
    size_t local_expert_start,
    size_t local_num_experts) {
    const auto shape = hidden_states->shape();
    INFINICORE_ASSERT(shape.size() == 2);
    auto output = Tensor::empty(
        shape, hidden_states->dtype(), hidden_states->device());
    moe_fused_dense_ascend_(
        output, hidden_states, w13, w2, topk_weights, topk_ids,
        global_num_experts, local_expert_start, local_num_experts);
    return output;
}

void moe_fused_dense_ascend_(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    size_t global_num_experts,
    size_t local_expert_start,
    size_t local_num_experts) {
    MoeFusedDenseAscend::execute(
        output, hidden_states, w13, w2, topk_weights, topk_ids,
        global_num_experts, local_expert_start, local_num_experts);
}

} // namespace infinicore::op
