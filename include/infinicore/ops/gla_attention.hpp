#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// Lightweight GLA-style attention built from existing primitives.
// Shapes:
//   q        : [B, n_q, S_q, D]
//   k_total  : [B, n_kv, S_kv, D]
//   v_total  : [B, n_kv, S_kv, D]
// Returns:
//   [B, n_q, S_q, D]
Tensor gla_attention(const Tensor &q,
                     const Tensor &k_total,
                     const Tensor &v_total,
                     float scale,
                     bool causal);

} // namespace infinicore::op

