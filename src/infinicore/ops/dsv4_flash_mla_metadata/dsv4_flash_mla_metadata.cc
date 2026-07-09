#include "infinicore/ops/dsv4_flash_mla_metadata.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4FlashMlaMetadata);

Dsv4FlashMlaMetadata::Dsv4FlashMlaMetadata(const Tensor &cache_seqlens, Tensor tile_scheduler_metadata, Tensor num_splits, int num_heads_per_head_k, int num_heads_k) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(cache_seqlens, tile_scheduler_metadata, num_splits);
    INFINICORE_GRAPH_OP_DISPATCH(cache_seqlens->device().getType(), cache_seqlens, tile_scheduler_metadata, num_splits, num_heads_per_head_k, num_heads_k);
}

void Dsv4FlashMlaMetadata::execute(const Tensor &cache_seqlens, Tensor tile_scheduler_metadata, Tensor num_splits, int num_heads_per_head_k, int num_heads_k) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4FlashMlaMetadata, cache_seqlens, tile_scheduler_metadata, num_splits, num_heads_per_head_k, num_heads_k);
}

void dsv4_flash_mla_metadata_(const Tensor &cache_seqlens, Tensor tile_scheduler_metadata, Tensor num_splits, int num_heads_per_head_k, int num_heads_k) {
    Dsv4FlashMlaMetadata::execute(cache_seqlens, tile_scheduler_metadata, num_splits, num_heads_per_head_k, num_heads_k);
}

} // namespace infinicore::op
