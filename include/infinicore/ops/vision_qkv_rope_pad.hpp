#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(VisionQKVRoPEPad, Tensor, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &);
__export void vision_qkv_rope_pad_(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &qkv, const Tensor &cos, const Tensor &sin, const Tensor &rows, const Tensor &cols);
} // namespace infinicore::op
