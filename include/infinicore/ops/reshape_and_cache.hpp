#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(ReshapeAndCache, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &,
                          const std::string &, Tensor &, Tensor &);

void reshape_and_cache(Tensor &key,          // [num_tokens, num_heads, head_size]
                       Tensor &value,        // [num_tokens, num_heads, head_size]
                       Tensor &key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
                       Tensor &value_cache,  // [num_blocks, num_heads, head_size, block_size]
                       Tensor &slot_mapping, // [num_tokens]
                       const std::string &kv_cache_dtype,
                       Tensor &k_scale,
                       Tensor &v_scale);

} // namespace infinicore::op
