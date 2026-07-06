#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(UnweightedRMSNorm, Tensor, const Tensor &, float);

Tensor unweighted_rms_norm(const Tensor &x, float epsilon = 1e-5f);
void unweighted_rms_norm_(Tensor y, const Tensor &x, float epsilon = 1e-5f);

} // namespace infinicore::op
