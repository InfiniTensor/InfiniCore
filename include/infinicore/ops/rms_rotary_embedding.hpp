#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <cstdint>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(RMSRotaryEmbedding,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          int64_t,
                          const Tensor &,
                          bool,
                          const Tensor &,
                          const Tensor &,
                          float);

bool rms_rotary_embedding_fuse_available(const Device &device);

void rms_rotary_embedding_fuse_(Tensor query,
                                Tensor key,
                                const Tensor &positions,
                                int64_t head_size,
                                const Tensor &cos_sin_cache,
                                bool is_neox,
                                const Tensor &q_weight,
                                const Tensor &k_weight,
                                float epsilon = 1e-6f);

} // namespace infinicore::op
