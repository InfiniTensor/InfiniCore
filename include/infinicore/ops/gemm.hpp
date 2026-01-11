#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Gemm, Tensor, Tensor, Tensor, float, float);

Tensor gemm(Tensor a, Tensor b, float alpha = 1.0f, float beta = 0.0f);
void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta);

} // namespace infinicore::op
