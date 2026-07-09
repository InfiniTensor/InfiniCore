#include "infinicore/ops/dsv4_flash_mla_decode.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4FlashMlaDecode);

Dsv4FlashMlaDecode::Dsv4FlashMlaDecode(Tensor output, Tensor softmax_lse, const Tensor &q_nope, const Tensor &q_pe, const Tensor &k_cache, const Tensor &block_table, const Tensor &cache_seqlens, const Tensor &tile_scheduler_metadata, const Tensor &num_splits, int head_dim_v, float softmax_scale, bool causal) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, softmax_lse, q_nope, q_pe, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, softmax_lse, q_nope, q_pe, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, head_dim_v, softmax_scale, causal);
}

void Dsv4FlashMlaDecode::execute(Tensor output, Tensor softmax_lse, const Tensor &q_nope, const Tensor &q_pe, const Tensor &k_cache, const Tensor &block_table, const Tensor &cache_seqlens, const Tensor &tile_scheduler_metadata, const Tensor &num_splits, int head_dim_v, float softmax_scale, bool causal) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4FlashMlaDecode, output, softmax_lse, q_nope, q_pe, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, head_dim_v, softmax_scale, causal);
}

void dsv4_flash_mla_decode_(Tensor output, Tensor softmax_lse, const Tensor &q_nope, const Tensor &q_pe, const Tensor &k_cache, const Tensor &block_table, const Tensor &cache_seqlens, const Tensor &tile_scheduler_metadata, const Tensor &num_splits, int head_dim_v, float softmax_scale, bool causal) {
    Dsv4FlashMlaDecode::execute(output, softmax_lse, q_nope, q_pe, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, head_dim_v, softmax_scale, causal);
}

} // namespace infinicore::op
