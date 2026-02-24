#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(PagedAttentionV1, Tensor &, Tensor &, Tensor &, Tensor &,
                          int64_t, double, Tensor &, Tensor &, int64_t, int64_t, const std::optional<Tensor> &,
                          const std::string &, Tensor &, Tensor &, int64_t, int64_t, int64_t, int64_t, int64_t);

void paged_attention_v1(Tensor &out,          // [num_seqs, num_heads, head_size]
                        Tensor &query,        // [num_seqs, num_heads, head_size]
                        Tensor &key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
                        Tensor &value_cache,  // [num_blocks, num_heads, head_size, block_size]
                        int64_t num_kv_heads, // [num_heads]
                        double scale,
                        Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
                        Tensor &seq_lens,     // [num_seqs]
                        int64_t block_size,
                        int64_t max_seq_len,
                        const std::optional<Tensor> &alibi_slopes,
                        const std::string &kv_cache_dtype, // "auto"
                        Tensor &k_scale,
                        Tensor &v_scale,
                        const int64_t tp_rank,
                        const int64_t blocksparse_local_blocks,
                        const int64_t blocksparse_vert_stride,
                        const int64_t blocksparse_block_size,
                        const int64_t blocksparse_head_sliding_step);

} // namespace infinicore::op
