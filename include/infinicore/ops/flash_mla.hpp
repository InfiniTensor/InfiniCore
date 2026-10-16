#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(FlashMlaSparsePrefill, Tensor, const Tensor &,
                          const Tensor &, const Tensor &, const Tensor &,
                          float);

INFINICORE_GRAPH_OP_CLASS(FlashMlaSparseDecode, Tensor, const Tensor &,
                          const Tensor &, const Tensor &, const Tensor &,
                          float);

void flash_mla_sparse_prefill_(Tensor output, const Tensor &query,
                               const Tensor &kv_cache, const Tensor &indices,
                               const Tensor &topk_lens, float scale);

void flash_mla_sparse_decode_(Tensor output, const Tensor &query,
                              const Tensor &kv_cache, const Tensor &indices,
                              const Tensor &topk_lens, float scale);

} // namespace infinicore::op
