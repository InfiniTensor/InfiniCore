#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Dsv4SwaPrefillIndices, Tensor, int);
void dsv4_swa_prefill_indices_(Tensor indices, int window_size);
} // namespace infinicore::op
