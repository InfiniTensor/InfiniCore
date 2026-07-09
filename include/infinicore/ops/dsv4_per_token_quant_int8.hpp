#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <utility>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4PerTokenQuantInt8, Tensor, Tensor, const Tensor &);

std::pair<Tensor, Tensor> dsv4_per_token_quant_int8(const Tensor &x);
void dsv4_per_token_quant_int8_(Tensor q, Tensor scale, const Tensor &x);

} // namespace infinicore::op
