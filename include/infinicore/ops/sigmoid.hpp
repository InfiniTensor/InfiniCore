#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Sigmoid, Tensor, const Tensor &);

__export Tensor sigmoid(const Tensor &input);
__export void sigmoid_(Tensor output, const Tensor &input);

} // namespace infinicore::op
