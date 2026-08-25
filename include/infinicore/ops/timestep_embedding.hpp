#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(TimestepEmbedding, Tensor, const Tensor &, float);

Tensor timestep_embedding(const Tensor &timestep,
                          size_t embedding_dim = 256,
                          float max_period = 10000.0f);

void timestep_embedding_(Tensor output,
                         const Tensor &timestep,
                         float max_period = 10000.0f);

} // namespace infinicore::op
