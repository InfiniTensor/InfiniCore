#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangMaskTopkIds,
                          Tensor,
                          const Tensor &);

void dsv4_sglang_mask_topk_ids_(Tensor topk_ids, const Tensor &num_token_non_padded);

} // namespace infinicore::op
