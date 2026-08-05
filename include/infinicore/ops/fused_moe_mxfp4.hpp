#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
#include "fused_moe.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(FusedMoeMxfp4,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          FusedMoeActivation);

Tensor fused_moe_mxfp4(const Tensor &input,
                       const Tensor &selected_experts,
                       const Tensor &routing_weights,
                       const Tensor &w13_packed,
                       const Tensor &w13_scale,
                       const Tensor &w2_packed,
                       const Tensor &w2_scale,
                       FusedMoeActivation activation);

void fused_moe_mxfp4_(Tensor output,
                      const Tensor &input,
                      const Tensor &selected_experts,
                      const Tensor &routing_weights,
                      const Tensor &w13_packed,
                      const Tensor &w13_scale,
                      const Tensor &w2_packed,
                      const Tensor &w2_scale,
                      FusedMoeActivation activation);

} // namespace infinicore::op
