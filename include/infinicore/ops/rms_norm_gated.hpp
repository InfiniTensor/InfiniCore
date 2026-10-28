#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
namespace infinicore::op {
// BF16 head-128 norm followed by SiLU(gate) * norm, with BF16 intermediates.
INFINICORE_GRAPH_OP_CLASS(RMSNormGated, Tensor, const Tensor &, const Tensor &, const Tensor &, float);
__export void rms_norm_gated_(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &weight, float epsilon);
__export Tensor rms_norm_gated(const Tensor &x, const Tensor &gate, const Tensor &weight, float epsilon);
} // namespace infinicore::op
