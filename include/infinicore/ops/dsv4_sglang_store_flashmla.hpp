#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Dsv4SglangStoreFlashmla,
                          const Tensor &, Tensor, const Tensor &);
void dsv4_sglang_store_flashmla_(const Tensor &input, Tensor cache, const Tensor &indices);
} // namespace infinicore::op
