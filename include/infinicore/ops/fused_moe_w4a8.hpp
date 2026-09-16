#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
#include "fused_moe.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(FusedMoeW4A8,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          FusedMoeActivation,
                          bool);

Tensor fused_moe_w4a8(const Tensor &input,
                      const Tensor &selected_experts,
                      const Tensor &routing_weights,
                      const Tensor &w13_packed,
                      const Tensor &w13_scale,
                      const Tensor &w2_packed,
                      const Tensor &w2_scale,
                      FusedMoeActivation activation,
                      bool weights_are_aiter_shuffled = false);

void fused_moe_w4a8_(Tensor output,
                     const Tensor &input,
                     const Tensor &selected_experts,
                     const Tensor &routing_weights,
                     const Tensor &w13_packed,
                     const Tensor &w13_scale,
                     const Tensor &w2_packed,
                     const Tensor &w2_scale,
                     FusedMoeActivation activation,
                     bool weights_are_aiter_shuffled = false);

} // namespace infinicore::op
