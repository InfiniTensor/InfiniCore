#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(KimiAttentionResidual,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float);

__export Tensor kimi_attention_residual(const Tensor &values,
                                        const Tensor &norm_weight,
                                        const Tensor &projection_weight,
                                        float epsilon = 1e-6f);

__export void kimi_attention_residual_(Tensor output,
                                       const Tensor &values,
                                       const Tensor &norm_weight,
                                       const Tensor &projection_weight,
                                       float epsilon = 1e-6f);

} // namespace infinicore::op
