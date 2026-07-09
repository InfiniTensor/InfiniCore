#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Dsv4SglangMegaMoePreDispatch,
                          const Tensor &, const Tensor &, const Tensor &, Tensor, Tensor, Tensor, Tensor);
void dsv4_sglang_mega_moe_pre_dispatch_(const Tensor &x, const Tensor &topk_idx, const Tensor &topk_weights, Tensor buf_x, Tensor buf_x_sf, Tensor buf_topk_idx, Tensor buf_topk_weights);
} // namespace infinicore::op
