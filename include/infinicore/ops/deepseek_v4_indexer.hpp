#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <cstddef>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4Indexer,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          size_t,
                          size_t);

Tensor deepseek_v4_indexer(const Tensor &q,
                           const Tensor &weights,
                           const Tensor &compressed,
                           const Tensor &positions,
                           size_t topk,
                           size_t query_start,
                           size_t compress_ratio);

void deepseek_v4_indexer_(Tensor indices,
                          const Tensor &q,
                          const Tensor &weights,
                          const Tensor &compressed,
                          const Tensor &positions,
                          size_t query_start,
                          size_t compress_ratio);

} // namespace infinicore::op
