#pragma once

#include "../tensor.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op {

// Converts packed per-channel W4 weights [E, N, K / 2] to the Marlin
// layout in a same-size output tensor. This is a one-time weight-loading op.
void prepare_w4a8_marlin_weight_(Tensor output, const Tensor &input);

// Pads an expert-sorted route permutation to block_size and emits one expert
// id per padded block. The output capacities are [routes * block_size] and
// [routes], while num_tokens_post_pad stores the used prefix length.
void moe_align_block_size_from_counts_(
    Tensor padded_sorted_token_ids,
    Tensor expert_ids,
    Tensor num_tokens_post_pad,
    const Tensor &sorted_token_ids,
    const Tensor &tokens_per_expert,
    int64_t block_size,
    int64_t routing_topk);

// Runs a per-channel W4A8 Marlin MoE GEMM. routing_topk describes the route
// permutation capacity and can differ from topk for the second expert GEMM.
void moe_w4a8_marlin_(
    Tensor output,
    const Tensor &input,
    const Tensor &marlin_weight,
    const Tensor &input_scale,
    const Tensor &weight_scale,
    std::optional<Tensor> topk_weights,
    const Tensor &padded_sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_pad,
    int64_t topk,
    int64_t routing_topk);

} // namespace infinicore::op
