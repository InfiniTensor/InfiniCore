#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MoeFusedDenseAscend,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          size_t,
                          size_t,
                          size_t);

Tensor moe_fused_dense_ascend(
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    size_t global_num_experts,
    size_t local_expert_start,
    size_t local_num_experts);

void moe_fused_dense_ascend_(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    size_t global_num_experts,
    size_t local_expert_start,
    size_t local_num_experts);

} // namespace infinicore::op
