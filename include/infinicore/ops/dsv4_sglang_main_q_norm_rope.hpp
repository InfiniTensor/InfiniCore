#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangMainQNormRope,
                          Tensor, const Tensor &, const Tensor &, const Tensor &, double);

void dsv4_sglang_main_q_norm_rope_(Tensor output, const Tensor &input, const Tensor &freqs, const Tensor &positions, double eps);

} // namespace infinicore::op
