#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Sigmoid, Tensor, Tensor);

Tensor sigmoid(Tensor input);
void sigmoid_(Tensor output, Tensor input);

} // namespace infinicore::op
