#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {
//
// INFINICORE_GRAPH_OP_CLASS(LinearW8A8I8, Tensor, const Tensor &, const Tensor &, std::optional<Tensor>);

Tensor linear_w8a8i8(Tensor input, Tensor weight_packed, Tensor weight_scale, std::optional<Tensor> bias);

void linear_w8a8i8_(Tensor out, Tensor input, Tensor weight_packed, Tensor weight_scale, std::optional<Tensor> bias);

} // namespace infinicore::op
