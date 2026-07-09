#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Dsv4AddRMSNormQuant, Tensor, Tensor, Tensor, const Tensor &, const Tensor &, float);
void dsv4_add_rmsnorm_quant_(Tensor res, Tensor q, Tensor scale, const Tensor &x, const Tensor &weight, float epsilon);
} // namespace infinicore::op
