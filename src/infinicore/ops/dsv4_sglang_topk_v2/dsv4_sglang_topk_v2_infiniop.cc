#include "infinicore/ops/dsv4_sglang_topk_v2.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_topk_v2.h"

namespace infinicore::op::dsv4_sglang_topk_v2_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangTopkV2, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, scores, seq_lens, page_table, page_indices, transform_workspace, metadata;
};

void *plan(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor transform_workspace, Tensor metadata, int64_t page_size) {
    size_t seed = hash_combine(scores, seq_lens, page_table, page_indices, transform_workspace, metadata, page_size);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangTopkV2,
        seed,
        scores->desc(), seq_lens->desc(), page_table->desc(), page_indices->desc(), transform_workspace->desc(), metadata->desc(), page_size);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangTopkV2, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(scores), graph::GraphTensor(seq_lens), graph::GraphTensor(page_table), graph::GraphTensor(page_indices), graph::GraphTensor(transform_workspace), graph::GraphTensor(metadata)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangTopkV2(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->scores->data(), planned->seq_lens->data(), planned->page_table->data(), planned->page_indices->data(), planned->transform_workspace->data(), planned->metadata->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangTopkV2, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_topk_v2_impl::infiniop
