#include "infinicore/ops/deepseek_v4_mhc_head.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::deepseek_v4_mhc_head_collapse_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4MHCHeadCollapse, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor y;
    graph::GraphTensor x;
    graph::GraphTensor mixes;
    graph::GraphTensor base;
    graph::GraphTensor scale;
};

void *plan(Tensor y,
           const Tensor &x,
           const Tensor &mixes,
           const Tensor &base,
           const Tensor &scale,
           float epsilon) {
    size_t seed = hash_combine(y, x, mixes, base, scale, epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4MHCHeadCollapse,
        seed,
        y->desc(),
        x->desc(),
        mixes->desc(),
        base->desc(),
        scale->desc(),
        epsilon);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4MHCHeadCollapse, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x),
        graph::GraphTensor(mixes),
        graph::GraphTensor(base),
        graph::GraphTensor(scale)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4MHCHeadCollapse(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->y->data(),
        planned->x->data(),
        planned->mixes->data(),
        planned->base->data(),
        planned->scale->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MHCHeadCollapse, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_mhc_head_collapse_impl::infiniop
