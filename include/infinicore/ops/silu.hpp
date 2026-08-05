#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Silu, Tensor, Tensor);

Tensor silu(Tensor input);
void silu_(Tensor output, Tensor input);
} // namespace infinicore::op
