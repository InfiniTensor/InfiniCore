#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SimpleGlaAttention,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float);

// Simple GLA (recurrent linear) attention with per-head decay.
// Shapes: q, k, v [B, T, H, D], g_gamma [H] (log-decay per head).
// Recurrence: gate = exp(g_gamma); S = S * gate + outer(k_t, v_t); o_t = (q_t * scale) @ S.
// Returns [B, T, H, D]. Implementation lives in InfiniOP (CPU reference; NVIDIA uses simple_gla_prefill kernels).
Tensor simple_gla_attention(const Tensor &q,
                            const Tensor &k,
                            const Tensor &v,
                            const Tensor &g_gamma,
                            float scale);

} // namespace infinicore::op
