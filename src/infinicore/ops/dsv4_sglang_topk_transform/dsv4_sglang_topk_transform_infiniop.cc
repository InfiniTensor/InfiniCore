#include "infinicore/ops/dsv4_sglang_topk_transform.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_topk_transform.h"

namespace infinicore::op::dsv4_sglang_topk_transform_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangTopkTransform, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, scores, seq_lens, page_table, page_indices, raw_indices;
};

void *plan(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor raw_indices, int64_t page_size) {
    size_t seed = hash_combine(scores, seq_lens, page_table, page_indices, raw_indices, page_size);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangTopkTransform,
        seed,
        scores->desc(), seq_lens->desc(), page_table->desc(), page_indices->desc(), raw_indices->desc(), page_size);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangTopkTransform, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(scores),
        graph::GraphTensor(seq_lens),
        graph::GraphTensor(page_table),
        graph::GraphTensor(page_indices),
        graph::GraphTensor(raw_indices)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangTopkTransform(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->scores->data(),
        planned->seq_lens->data(),
        planned->page_table->data(),
        planned->page_indices->data(),
        planned->raw_indices->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangTopkTransform, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_topk_transform_impl::infiniop
