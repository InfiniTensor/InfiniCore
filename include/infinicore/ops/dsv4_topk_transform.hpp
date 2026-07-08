#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4TopkTransform, Tensor, const Tensor &, const Tensor &, const Tensor &, int);

Tensor dsv4_topk_transform(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_tables, int page_size = 64);
void dsv4_topk_transform_(Tensor out, const Tensor &scores, const Tensor &seq_lens, const Tensor &page_tables, int page_size = 64);

} // namespace infinicore::op
