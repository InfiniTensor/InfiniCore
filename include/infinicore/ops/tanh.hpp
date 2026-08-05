#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Tanh, Tensor, const Tensor &);

__export Tensor tanh(const Tensor &input);
__export void tanh_(Tensor output, const Tensor &input);

} // namespace infinicore::op
