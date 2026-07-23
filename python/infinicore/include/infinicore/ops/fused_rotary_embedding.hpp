#pragma once

#include "../tensor.hpp"

namespace infinicore::op {

// Apply rotary embedding to query and key in one vendor-kernel launch. This
// mirrors vLLM's rotary_embedding(positions, query, key, ...) execution path.
void fused_rotary_embedding_(Tensor query,
                             Tensor key,
                             const Tensor &positions,
                             int64_t head_size,
                             const Tensor &cos_sin_cache,
                             bool is_neox);

} // namespace infinicore::op
