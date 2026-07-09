#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SiluAndMul, Tensor, const Tensor &, const Tensor &);

Tensor dsv4_silu_and_mul(const Tensor &gate, const Tensor &up);
void dsv4_silu_and_mul_(Tensor y, const Tensor &gate, const Tensor &up);

} // namespace infinicore::op
