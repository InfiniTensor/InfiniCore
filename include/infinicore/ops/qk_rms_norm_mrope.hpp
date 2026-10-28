#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(QKRMSNormMRoPE, Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &, float, int, int, int, bool);
__export void qk_rms_norm_mrope_(Tensor q_out,
                                 Tensor k_out,
                                 const Tensor &q,
                                 const Tensor &k,
                                 const Tensor &q_weight,
                                 const Tensor &k_weight,
                                 const Tensor &cos,
                                 const Tensor &sin,
                                 const Tensor &positions, float epsilon, int section_t, int section_h, int section_w, bool interleaved);
} // namespace infinicore::op
