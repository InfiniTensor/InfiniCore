#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangSiluAndMulClamp,
                          Tensor, const Tensor &, double);

void dsv4_sglang_silu_and_mul_clamp_(Tensor output, const Tensor &input, double limit);

} // namespace infinicore::op
