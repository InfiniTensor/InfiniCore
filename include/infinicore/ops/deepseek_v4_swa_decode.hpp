#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4SwaDecode,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float);

Tensor deepseek_v4_swa_decode(const Tensor &q,
                              const Tensor &k,
                              const Tensor &attn_sink,
                              float softmax_scale);

void deepseek_v4_swa_decode_(Tensor y,
                             const Tensor &q,
                             const Tensor &k,
                             const Tensor &attn_sink,
                             float softmax_scale);

} // namespace infinicore::op
