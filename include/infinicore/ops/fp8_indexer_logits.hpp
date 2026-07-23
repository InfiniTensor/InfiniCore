#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    Fp8IndexerLogits,
    Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &,
    const Tensor &, const Tensor &);

void fp8_indexer_logits_(
    Tensor logits,
    const Tensor &q_fp8,
    const Tensor &kv_cache,
    const Tensor &block_tables,
    const Tensor &weights_fp32,
    const Tensor &positions,
    const Tensor &request_ids);

} // namespace infinicore::op
