#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(GDNQKVNorm, Tensor, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &);
__export void gdn_qkv_norm_(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &q, const Tensor &k, const Tensor &v);
} // namespace infinicore::op
