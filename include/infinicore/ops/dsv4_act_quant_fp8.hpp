#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Dsv4ActQuantFp8, Tensor, Tensor, const Tensor &, float);
void dsv4_act_quant_fp8_(Tensor xq, Tensor scale, const Tensor &x, float fp8_max);
} // namespace infinicore::op
