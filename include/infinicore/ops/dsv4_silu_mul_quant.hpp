#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Dsv4SiluMulQuant, Tensor, Tensor, const Tensor &, const Tensor &);
void dsv4_silu_mul_quant_(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up);
} // namespace infinicore::op
