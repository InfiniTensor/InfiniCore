#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SimpleGLAPrefill,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float);

// Fused/chunked Simple GLA prefill forward.
// q,k,v: [B,T,H,D] (F16/BF16), g_gamma: [H] (F32), returns [B,T,H,D] (same dtype).
Tensor simple_gla_prefill(const Tensor &q,
                          const Tensor &k,
                          const Tensor &v,
                          const Tensor &g_gamma,
                          float scale);

} // namespace infinicore::op

