#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Dsv4SglangCompressC128OnlineV2Prefill,
                          const Tensor &, const Tensor &, Tensor, const Tensor &, const Tensor &, const Tensor &);
void dsv4_sglang_compress_c128_online_v2_prefill_(const Tensor &kv_score_buffer, const Tensor &kv_score_input, Tensor kv_output, const Tensor &ape, const Tensor &plan_c, const Tensor &plan_w);
} // namespace infinicore::op
