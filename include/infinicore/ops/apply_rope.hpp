#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(ApplyRoPE, Tensor, Tensor, const Tensor &, const Tensor &);
// In-place NeoX rotation of contiguous TND Q/K using full-width cos/sin.
void apply_rope_(Tensor q, Tensor k, const Tensor &cos, const Tensor &sin);
} // namespace infinicore::op
