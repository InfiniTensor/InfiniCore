#pragma once

#include <cstdint>

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangTopkV2,
                          const Tensor &, const Tensor &, const Tensor &, Tensor, Tensor, Tensor, int64_t);

void dsv4_sglang_topk_v2_(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor transform_workspace, Tensor metadata, int64_t page_size);

} // namespace infinicore::op
