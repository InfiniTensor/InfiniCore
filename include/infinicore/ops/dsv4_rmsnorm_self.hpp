#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4RMSNormSelf, Tensor, const Tensor &, float);

Tensor dsv4_rmsnorm_self(const Tensor &x, float epsilon = 1e-6f);
void dsv4_rmsnorm_self_(Tensor y, const Tensor &x, float epsilon = 1e-6f);

} // namespace infinicore::op
