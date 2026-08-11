#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <cstddef>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MoeW16A16MarlinFusedDense,
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
                          size_t,
                          int,
                          int,
                          int,
                          int);

Tensor moe_w16a16_marlin_pack(const Tensor &weight);

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
    int delta1);

} // namespace infinicore::op
