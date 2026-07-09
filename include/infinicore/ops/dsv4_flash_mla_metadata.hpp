#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4FlashMlaMetadata,
                          const Tensor &,
                          Tensor,
                          Tensor,
                          int,
                          int);

void dsv4_flash_mla_metadata_(const Tensor &cache_seqlens, Tensor tile_scheduler_metadata, Tensor num_splits, int num_heads_per_head_k, int num_heads_k);

} // namespace infinicore::op
