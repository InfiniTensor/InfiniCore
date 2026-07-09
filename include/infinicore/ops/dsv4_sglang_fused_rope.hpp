#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangFusedRope,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          bool);

void dsv4_sglang_fused_rope_(Tensor q, const Tensor &freqs_cis, const Tensor &positions, bool inverse = false);

} // namespace infinicore::op
