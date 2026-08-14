#include "infinicore/ops/moe_w16a16_marlin.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

namespace moe_w16a16_marlin_pack_impl {
using schema = Tensor (*)(const Tensor &);
common::OpDispatcher<schema> &dispatcher() {
    static common::OpDispatcher<schema> dispatcher_;
    return dispatcher_;
}
} // namespace moe_w16a16_marlin_pack_impl

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeW16A16MarlinFusedDense);

MoeW16A16MarlinFusedDense::MoeW16A16MarlinFusedDense(
    Tensor output,
    Tensor cache13,
    Tensor cache2,
    const Tensor &hidden_states,
    const Tensor &w13_marlin,
    const Tensor &w2_marlin,
    const Tensor &topk_weights,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded,
    size_t top_k,
    int mode0,
    int delta0,
    int mode1,
    int delta1) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, cache13, cache2, hidden_states, w13_marlin, w2_marlin,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, cache13, cache2, hidden_states,
        w13_marlin, w2_marlin, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, top_k, mode0, delta0, mode1, delta1);
}

void MoeW16A16MarlinFusedDense::execute(
    Tensor output,
    Tensor cache13,
    Tensor cache2,
    const Tensor &hidden_states,
    const Tensor &w13_marlin,
    const Tensor &w2_marlin,
    const Tensor &topk_weights,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded,
    size_t top_k,
    int mode0,
    int delta0,
    int mode1,
    int delta1) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeW16A16MarlinFusedDense, output, cache13, cache2, hidden_states,
        w13_marlin, w2_marlin, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, top_k, mode0, delta0, mode1, delta1);
}

Tensor moe_w16a16_marlin_pack(const Tensor &weight) {
    return moe_w16a16_marlin_pack_impl::dispatcher().lookup(weight->device().getType())(weight);
}

void moe_w16a16_marlin_fused_dense_(
    Tensor output,
    Tensor cache13,
    Tensor cache2,
    const Tensor &hidden_states,
    const Tensor &w13_marlin,
    const Tensor &w2_marlin,
    const Tensor &topk_weights,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded,
    size_t top_k,
    int mode0,
    int delta0,
    int mode1,
    int delta1) {
    MoeW16A16MarlinFusedDense::execute(
        output, cache13, cache2, hidden_states, w13_marlin, w2_marlin,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k, mode0, delta0, mode1, delta1);
}

} // namespace infinicore::op
