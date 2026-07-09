#include "infinicore/ops/dsv4_flash_mla_decode.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_flash_mla_decode.h"

namespace infinicore::op::dsv4_flash_mla_decode_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4FlashMlaDecode, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, softmax_lse, q_nope, q_pe, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits;
};

void *plan(Tensor output, Tensor softmax_lse, const Tensor &q_nope, const Tensor &q_pe, const Tensor &k_cache, const Tensor &block_table, const Tensor &cache_seqlens, const Tensor &tile_scheduler_metadata, const Tensor &num_splits, int head_dim_v, float softmax_scale, bool causal) {
    size_t seed = hash_combine(output, softmax_lse, q_nope, q_pe, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, head_dim_v, softmax_scale, causal);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4FlashMlaDecode,
        seed,
        output->desc(), softmax_lse->desc(), q_nope->desc(), q_pe->desc(), k_cache->desc(), block_table->desc(), cache_seqlens->desc(), tile_scheduler_metadata->desc(), num_splits->desc(), head_dim_v, softmax_scale, causal);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4FlashMlaDecode, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(softmax_lse),
        graph::GraphTensor(q_nope),
        graph::GraphTensor(q_pe),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(block_table),
        graph::GraphTensor(cache_seqlens),
        graph::GraphTensor(tile_scheduler_metadata),
        graph::GraphTensor(num_splits)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4FlashMlaDecode(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->softmax_lse->data(),
        planned->q_nope->data(),
        planned->q_pe->data(),
        planned->k_cache->data(),
        planned->block_table->data(),
        planned->cache_seqlens->data(),
        planned->tile_scheduler_metadata->data(),
        planned->num_splits->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4FlashMlaDecode, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_flash_mla_decode_impl::infiniop
