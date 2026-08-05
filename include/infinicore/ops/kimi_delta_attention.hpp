#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(KimiDeltaAttention,
                          Tensor,
                          Tensor,
                          std::optional<Tensor>,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          float,
                          float,
                          bool);

__export Tensor kimi_delta_attention(const Tensor &q,
                                     const Tensor &k,
                                     const Tensor &v,
                                     const Tensor &g,
                                     const Tensor &beta,
                                     const Tensor &A_log,
                                     const Tensor &dt_bias,
                                     Tensor initial_state,
                                     std::optional<Tensor> cu_seqlens = std::nullopt,
                                     std::optional<Tensor> initial_state_indices = std::nullopt,
                                     std::optional<Tensor> final_state_indices = std::nullopt,
                                     float scale = 1.0f,
                                     float lower_bound = -5.0f,
                                     bool use_qk_l2norm = true);

__export void kimi_delta_attention_(Tensor out,
                                    Tensor initial_state,
                                    std::optional<Tensor> final_state,
                                    const Tensor &q,
                                    const Tensor &k,
                                    const Tensor &v,
                                    const Tensor &g,
                                    const Tensor &beta,
                                    const Tensor &A_log,
                                    const Tensor &dt_bias,
                                    std::optional<Tensor> cu_seqlens,
                                    std::optional<Tensor> initial_state_indices,
                                    std::optional<Tensor> final_state_indices,
                                    float scale = 1.0f,
                                    float lower_bound = -5.0f,
                                    bool use_qk_l2norm = true);

} // namespace infinicore::op
