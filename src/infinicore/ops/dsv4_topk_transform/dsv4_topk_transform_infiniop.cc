#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_topk_transform.hpp"
#include "infiniop/ops/dsv4_topk_transform.h"

namespace infinicore::op::dsv4_topk_transform_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4TopkTransform, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, scores, seq_lens, page_tables;
};

void *plan(Tensor out, const Tensor &scores, const Tensor &seq_lens, const Tensor &page_tables, int page_size) {
    size_t seed = hash_combine(out, scores, seq_lens, page_tables, page_size);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, Dsv4TopkTransform, seed, out->desc(), scores->desc(), seq_lens->desc(), page_tables->desc(), page_size);
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4TopkTransform, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(out), graph::GraphTensor(scores), graph::GraphTensor(seq_lens), graph::GraphTensor(page_tables)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4TopkTransform(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->out->data(), p->scores->data(), p->seq_lens->data(), p->page_tables->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4TopkTransform, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_topk_transform_impl::infiniop
