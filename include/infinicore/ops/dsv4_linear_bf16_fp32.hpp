#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4LinearBf16Fp32, Tensor, const Tensor &, const Tensor &);

Tensor dsv4_linear_bf16_fp32(const Tensor &x, const Tensor &w);
void dsv4_linear_bf16_fp32_(Tensor y, const Tensor &x, const Tensor &w);

} // namespace infinicore::op
