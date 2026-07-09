#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4PerTokenGroupQuantInt8,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          int);

void dsv4_per_token_group_quant_int8_(Tensor q, Tensor scale, const Tensor &x, int group_size);

} // namespace infinicore::op
