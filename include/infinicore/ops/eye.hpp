#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Eye, Tensor);

Tensor eye(size_t n, std::optional<size_t> m, const DataType &dtype, const Device &device);
void eye_(Tensor y);

} // namespace infinicore::op
