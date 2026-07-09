#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangFusedNormRope,
                          Tensor, const Tensor &, const Tensor &, const Tensor &, double);

void dsv4_sglang_fused_norm_rope_(Tensor kv, const Tensor &weight, const Tensor &positions, const Tensor &freqs, double eps);

} // namespace infinicore::op
