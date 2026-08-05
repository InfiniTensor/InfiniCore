#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    Fp8IndexerQuant, Tensor, Tensor, const Tensor &, const Tensor &);

void fp8_indexer_quant_(
    Tensor q_fp8,
    Tensor weights_fp32,
    const Tensor &q,
    const Tensor &weights);

INFINICORE_GRAPH_OP_CLASS(
    FusedFp8Indexer,
    Tensor, Tensor, Tensor,
    const Tensor &, const Tensor &, const Tensor &, const Tensor &,
    const Tensor &, const Tensor &, const Tensor &,
    size_t, double, double);

void fused_fp8_indexer_(
    Tensor q_fp8,
    Tensor weights_fp32,
    Tensor k_cache,
    const Tensor &q_raw,
    const Tensor &k_weights,
    const Tensor &norm_weight,
    const Tensor &norm_bias,
    const Tensor &positions,
    const Tensor &cos_sin_cache,
    const Tensor &slot_mapping,
    size_t rope_dim,
    double eps,
    double weights_scale);

} // namespace infinicore::op
