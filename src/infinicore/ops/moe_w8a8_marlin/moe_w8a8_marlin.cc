#include "infinicore/ops/moe_w8a8_marlin.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

namespace moe_w8a8_marlin_pack_impl {
using schema = Tensor (*)(const Tensor &);
common::OpDispatcher<schema> &dispatcher() {
    static common::OpDispatcher<schema> dispatcher_;
    return dispatcher_;
}
} // namespace moe_w8a8_marlin_pack_impl

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeW8A8MarlinFusedDense);

MoeW8A8MarlinFusedDense::MoeW8A8MarlinFusedDense(
    Tensor output,
    Tensor cache13,
    Tensor cache2_i8,
    Tensor input_i8,
    Tensor input_scale,
    Tensor cache2_scale,
    const Tensor &hidden_states,
    const Tensor &w13_marlin,
    const Tensor &w2_marlin,
    const Tensor &w13_scale,
    const Tensor &w2_scale,
    const Tensor &topk_weights,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded,
    size_t top_k,
    int mode0,
    size_t block_size_m,
    int delta0,
    int mode1,
    int delta1) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, cache13, cache2_i8, input_i8, input_scale, cache2_scale,
        hidden_states, w13_marlin, w2_marlin, w13_scale, w2_scale,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(),
        output,
        cache13,
        cache2_i8,
        input_i8,
        input_scale,
        cache2_scale,
        hidden_states,
        w13_marlin,
        w2_marlin,
        w13_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        mode0,
        block_size_m,
        delta0,
        mode1,
        delta1);
}

void MoeW8A8MarlinFusedDense::execute(
    Tensor output,
    Tensor cache13,
    Tensor cache2_i8,
    Tensor input_i8,
    Tensor input_scale,
    Tensor cache2_scale,
    const Tensor &hidden_states,
    const Tensor &w13_marlin,
    const Tensor &w2_marlin,
    const Tensor &w13_scale,
    const Tensor &w2_scale,
    const Tensor &topk_weights,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded,
    size_t top_k,
    int mode0,
    size_t block_size_m,
    int delta0,
    int mode1,
    int delta1) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeW8A8MarlinFusedDense,
        output,
        cache13,
        cache2_i8,
        input_i8,
        input_scale,
        cache2_scale,
        hidden_states,
        w13_marlin,
        w2_marlin,
        w13_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        mode0,
        block_size_m,
        delta0,
        mode1,
        delta1);
}


Tensor moe_w8a8_marlin_pack(const Tensor &weight) {
    return moe_w8a8_marlin_pack_impl::dispatcher().lookup(weight->device().getType())(weight);
}

void moe_w8a8_marlin_fused_dense_(
    Tensor output,
    Tensor cache13,
    Tensor cache2_i8,
    Tensor input_i8,
    Tensor input_scale,
    Tensor cache2_scale,
    const Tensor &hidden_states,
    const Tensor &w13_marlin,
    const Tensor &w2_marlin,
    const Tensor &w13_scale,
    const Tensor &w2_scale,
    const Tensor &topk_weights,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded,
    size_t top_k,
    int mode0,
    size_t block_size_m,
    int delta0,
    int mode1,
    int delta1) {
    MoeW8A8MarlinFusedDense::execute(
        output, cache13, cache2_i8, input_i8, input_scale, cache2_scale,
        hidden_states, w13_marlin, w2_marlin, w13_scale, w2_scale,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k, mode0, block_size_m, delta0, mode1, delta1);
}


} // namespace infinicore::op
