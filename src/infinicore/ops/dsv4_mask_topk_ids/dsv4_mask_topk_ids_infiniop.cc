#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_mask_topk_ids.hpp"
#include "infiniop/ops/dsv4_mask_topk_ids.h"

namespace infinicore::op::dsv4_mask_topk_ids_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4MaskTopkIds, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, topk_ids, num_token_non_padded;
};

void *plan(Tensor topk_ids, const Tensor &num_token_non_padded) {
    size_t seed = hash_combine(topk_ids, num_token_non_padded);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4MaskTopkIds, seed, topk_ids->desc(), num_token_non_padded->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4MaskTopkIds, descriptor);
    return new PlannedMeta{
        descriptor, graph::GraphTensor(workspace), graph::GraphTensor(topk_ids), graph::GraphTensor(num_token_non_padded)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4MaskTopkIds(
        p->descriptor->desc,
        p->workspace->data(),
        p->workspace->numel(),
        p->topk_ids->data(),
        p->num_token_non_padded->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4MaskTopkIds, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_mask_topk_ids_impl::infiniop
