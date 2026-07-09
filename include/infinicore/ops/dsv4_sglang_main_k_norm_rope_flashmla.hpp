#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Dsv4SglangMainKNormRopeFlashmla,
                          Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor, double);
void dsv4_sglang_main_k_norm_rope_flashmla_(Tensor kv, const Tensor &weight, const Tensor &freqs, const Tensor &positions, const Tensor &out_loc, Tensor cache, double eps);
} // namespace infinicore::op
