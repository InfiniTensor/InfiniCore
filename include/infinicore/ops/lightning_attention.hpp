#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(LightningAttention,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

// Indexed-pool lightning attention (MiniMax-01 style).
// Returns out [B, T, H, D] and updates `initial_state` in place at
// `final_state_indices` rows.
__export Tensor lightning_attention(const Tensor &q,
                                    const Tensor &k,
                                    const Tensor &v,
                                    const Tensor &slope,
                                    Tensor initial_state,
                                    const Tensor &initial_state_indices,
                                    const Tensor &final_state_indices);

__export void lightning_attention_(Tensor out,
                                   Tensor initial_state,
                                   const Tensor &q,
                                   const Tensor &k,
                                   const Tensor &v,
                                   const Tensor &slope,
                                   const Tensor &initial_state_indices,
                                   const Tensor &final_state_indices);

} // namespace infinicore::op
