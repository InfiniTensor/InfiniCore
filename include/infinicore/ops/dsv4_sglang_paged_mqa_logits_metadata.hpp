#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SglangPagedMqaLogitsMetadata,
                          const Tensor &, Tensor);

void dsv4_sglang_paged_mqa_logits_metadata_(const Tensor &seq_lens, Tensor metadata);

} // namespace infinicore::op
