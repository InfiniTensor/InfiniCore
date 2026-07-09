#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangHashTopk,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          Tensor,
                          float);

void dsv4_sglang_hash_topk_(const Tensor &router_logits, const Tensor &input_ids, const Tensor &tid2eid, Tensor topk_weights, Tensor topk_ids, float routed_scaling_factor);

} // namespace infinicore::op
