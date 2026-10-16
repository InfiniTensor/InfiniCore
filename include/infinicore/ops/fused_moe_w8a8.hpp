#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(FusedMoeW8A8,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

Tensor fused_moe_w8a8(const Tensor &input,
                      const Tensor &selected_experts,
                      const Tensor &routing_weights,
                      const Tensor &w1,
                      const Tensor &w1_scale,
                      const Tensor &w2,
                      const Tensor &w2_scale);

void fused_moe_w8a8_(Tensor output,
                     const Tensor &input,
                     const Tensor &selected_experts,
                     const Tensor &routing_weights,
                     const Tensor &w1,
                     const Tensor &w1_scale,
                     const Tensor &w2,
                     const Tensor &w2_scale);

} // namespace infinicore::op
