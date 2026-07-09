#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4FusedRope, Tensor, Tensor, const Tensor &, const Tensor &, bool);

void dsv4_fused_rope_(Tensor q, Tensor k, const Tensor &freq_real, const Tensor &freq_imag, bool has_k = false);
Tensor dsv4_fused_rope(const Tensor &q, const Tensor &freq_real, const Tensor &freq_imag);

} // namespace infinicore::op
