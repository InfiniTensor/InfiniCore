#include "infinicore/ops/dsv4_flash_mla_metadata.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_flash_mla_metadata.h"

namespace infinicore::op::dsv4_flash_mla_metadata_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4FlashMlaMetadata, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, cache_seqlens, tile_scheduler_metadata, num_splits;
};

void *plan(const Tensor &cache_seqlens, Tensor tile_scheduler_metadata, Tensor num_splits, int num_heads_per_head_k, int num_heads_k) {
    size_t seed = hash_combine(cache_seqlens, tile_scheduler_metadata, num_splits, num_heads_per_head_k, num_heads_k);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4FlashMlaMetadata,
        seed,
        cache_seqlens->desc(), tile_scheduler_metadata->desc(), num_splits->desc(), num_heads_per_head_k, num_heads_k);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4FlashMlaMetadata, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(cache_seqlens),
        graph::GraphTensor(tile_scheduler_metadata),
        graph::GraphTensor(num_splits)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4FlashMlaMetadata(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->cache_seqlens->data(),
        planned->tile_scheduler_metadata->data(),
        planned->num_splits->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4FlashMlaMetadata, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_flash_mla_metadata_impl::infiniop
