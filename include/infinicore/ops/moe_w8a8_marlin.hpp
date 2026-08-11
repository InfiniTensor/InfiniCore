#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <cstddef>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MoeW8A8MarlinFusedDense,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          size_t,
                          int,
                          size_t,
                          int,
                          int,
                          int);

Tensor moe_w8a8_marlin_pack(const Tensor &weight);

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
    int delta1);

} // namespace infinicore::op
