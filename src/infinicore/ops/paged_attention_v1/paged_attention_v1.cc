#include "infinicore/ops/paged_attention_v1.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PagedAttentionV1);

PagedAttentionV1::PagedAttentionV1(Tensor &out,          // [num_seqs, num_heads, head_size]
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
                                   const int64_t blocksparse_head_sliding_step) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, query, key_cache, value_cache, block_tables, seq_lens);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, query, key_cache, value_cache,
                                 num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
                                 alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank,
                                 blocksparse_local_blocks, blocksparse_vert_stride,
                                 blocksparse_block_size, blocksparse_head_sliding_step);
}

void PagedAttentionV1::execute(Tensor &out,          // [num_seqs, num_heads, head_size]
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
                               const int64_t blocksparse_head_sliding_step) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        PagedAttentionV1,
        out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step);
}

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
                        const int64_t blocksparse_head_sliding_step) {

    PagedAttentionV1::execute(out, query, key_cache, value_cache,
                              num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
                              alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank,
                              blocksparse_local_blocks, blocksparse_vert_stride,
                              blocksparse_block_size, blocksparse_head_sliding_step);
}

} // namespace infinicore::op
