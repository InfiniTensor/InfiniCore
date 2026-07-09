#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangTopkTransform,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          Tensor,
                          int64_t);

void dsv4_sglang_topk_transform_(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor raw_indices, int64_t page_size);

} // namespace infinicore::op
