#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(W4A8MoeShuffle,
                          Tensor,
                          const Tensor &);

Tensor w4a8_moe_shuffle(const Tensor &input);
void w4a8_moe_shuffle_(Tensor output, const Tensor &input);

} // namespace infinicore::op
