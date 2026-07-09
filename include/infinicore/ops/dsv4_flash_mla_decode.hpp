#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4FlashMlaDecode,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          int,
                          float,
                          bool);

void dsv4_flash_mla_decode_(Tensor output,
                            Tensor softmax_lse,
                            const Tensor &q_nope,
                            const Tensor &q_pe,
                            const Tensor &k_cache,
                            const Tensor &block_table,
                            const Tensor &cache_seqlens,
                            const Tensor &tile_scheduler_metadata,
                            const Tensor &num_splits,
                            int head_dim_v,
                            float softmax_scale,
                            bool causal);

} // namespace infinicore::op
