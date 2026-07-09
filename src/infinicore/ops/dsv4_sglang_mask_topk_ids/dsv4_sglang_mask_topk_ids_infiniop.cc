#include "infinicore/ops/dsv4_sglang_mask_topk_ids.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_mask_topk_ids.h"

namespace infinicore::op::dsv4_sglang_mask_topk_ids_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangMaskTopkIds, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, topk_ids, num_token_non_padded;
};

void *plan(Tensor topk_ids, const Tensor &num_token_non_padded) {
    size_t seed = hash_combine(topk_ids, num_token_non_padded);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangMaskTopkIds,
        seed,
        topk_ids->desc(), num_token_non_padded->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangMaskTopkIds, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(topk_ids),
        graph::GraphTensor(num_token_non_padded)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangMaskTopkIds(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->topk_ids->data(),
        planned->num_token_non_padded->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangMaskTopkIds, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_mask_topk_ids_impl::infiniop
