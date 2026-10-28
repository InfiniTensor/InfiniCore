#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(QKRMSNormRoPE, Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &, float);
__export void qk_rms_norm_rope_(Tensor q_out,
                                Tensor k_out,
                                const Tensor &q,
                                const Tensor &k,
                                const Tensor &q_weight,
                                const Tensor &k_weight,
                                const Tensor &cos,
                                const Tensor &sin,
                                const Tensor &positions, float epsilon);
} // namespace infinicore::op
