#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    Fp8SparseMla,
    Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float);

void fp8_sparse_mla_(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale);

} // namespace infinicore::op
