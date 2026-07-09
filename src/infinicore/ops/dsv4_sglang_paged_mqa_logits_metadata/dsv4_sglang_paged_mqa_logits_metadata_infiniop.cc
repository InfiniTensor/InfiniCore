#include "infinicore/ops/dsv4_sglang_paged_mqa_logits_metadata.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_paged_mqa_logits_metadata.h"

namespace infinicore::op::dsv4_sglang_paged_mqa_logits_metadata_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangPagedMqaLogitsMetadata, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, seq_lens, metadata;
};

void *plan(const Tensor &seq_lens, Tensor metadata) {
    size_t seed = hash_combine(seq_lens, metadata);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangPagedMqaLogitsMetadata,
        seed,
        seq_lens->desc(), metadata->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangPagedMqaLogitsMetadata, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(seq_lens), graph::GraphTensor(metadata)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangPagedMqaLogitsMetadata(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->seq_lens->data(), planned->metadata->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangPagedMqaLogitsMetadata, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_paged_mqa_logits_metadata_impl::infiniop
