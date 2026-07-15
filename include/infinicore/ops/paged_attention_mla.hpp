#pragma once

#include "../tensor.hpp"

#include <cstdint>

namespace infinicore::op {

void paged_attention_mla_(Tensor output,
                          const Tensor &query,
                          const Tensor &kv_cache,
                          float scale,
                          const Tensor &block_tables,
                          const Tensor &context_lens,
                          int64_t max_context_len);

} // namespace infinicore::op
