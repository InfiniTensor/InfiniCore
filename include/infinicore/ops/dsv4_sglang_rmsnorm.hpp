#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangRmsnorm,
                          Tensor,
                          const Tensor &,
                          double);

void dsv4_sglang_rmsnorm_(Tensor output, const Tensor &input, double eps = 1e-6);

} // namespace infinicore::op
