#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SelectLastTokenHidden, Tensor, const Tensor &, const Tensor &);

void select_last_token_hidden_(Tensor output, const Tensor &hidden_states, const Tensor &input_offsets);

} // namespace infinicore::op
