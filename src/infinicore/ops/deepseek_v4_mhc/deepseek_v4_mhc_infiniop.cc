#include "infinicore/ops/deepseek_v4_mhc.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::deepseek_v4_mhc_params_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4MHCParams, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor pre;
    graph::GraphTensor post;
    graph::GraphTensor comb;
    graph::GraphTensor mixes;
    graph::GraphTensor base;
    graph::GraphTensor scale;
};

void *plan(Tensor pre,
           Tensor post,
           Tensor comb,
           const Tensor &mixes,
           const Tensor &base,
           const Tensor &scale,
           size_t sinkhorn_iters,
           float epsilon) {
    size_t seed = hash_combine(pre, post, comb, mixes, base, scale, sinkhorn_iters, epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4MHCParams,
        seed,
        pre->desc(),
        post->desc(),
        comb->desc(),
        mixes->desc(),
        base->desc(),
        scale->desc(),
        sinkhorn_iters,
        epsilon);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4MHCParams, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(pre),
        graph::GraphTensor(post),
        graph::GraphTensor(comb),
        graph::GraphTensor(mixes),
        graph::GraphTensor(base),
        graph::GraphTensor(scale)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4MHCParams(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->pre->data(),
        planned->post->data(),
        planned->comb->data(),
        planned->mixes->data(),
        planned->base->data(),
        planned->scale->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MHCParams, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_mhc_params_impl::infiniop


namespace infinicore::op::deepseek_v4_mhc_pre_collapse_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4MHCPreCollapse, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor collapsed;
    graph::GraphTensor post;
    graph::GraphTensor comb;
    graph::GraphTensor x;
    graph::GraphTensor mixes;
    graph::GraphTensor base;
    graph::GraphTensor scale;
};

void *plan(Tensor collapsed,
           Tensor post,
           Tensor comb,
           const Tensor &x,
           const Tensor &mixes,
           const Tensor &base,
           const Tensor &scale,
           size_t sinkhorn_iters,
           float epsilon) {
    size_t seed = hash_combine(collapsed, post, comb, x, mixes, base, scale, sinkhorn_iters, epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4MHCPreCollapse,
        seed,
        collapsed->desc(),
        post->desc(),
        comb->desc(),
        x->desc(),
        mixes->desc(),
        base->desc(),
        scale->desc(),
        sinkhorn_iters,
        epsilon);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4MHCPreCollapse, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(collapsed),
        graph::GraphTensor(post),
        graph::GraphTensor(comb),
        graph::GraphTensor(x),
        graph::GraphTensor(mixes),
        graph::GraphTensor(base),
        graph::GraphTensor(scale)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4MHCPreCollapse(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->collapsed->data(),
        planned->post->data(),
        planned->comb->data(),
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

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MHCPreCollapse, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_mhc_pre_collapse_impl::infiniop


namespace infinicore::op::deepseek_v4_mhc_scale_mixes_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4MHCScaleMixes, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor scaled;
    graph::GraphTensor x;
    graph::GraphTensor raw_mixes;
};

void *plan(Tensor scaled,
           const Tensor &x,
           const Tensor &raw_mixes,
           float epsilon) {
    size_t seed = hash_combine(scaled, x, raw_mixes, epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4MHCScaleMixes,
        seed,
        scaled->desc(),
        x->desc(),
        raw_mixes->desc(),
        epsilon);
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4MHCScaleMixes, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(scaled),
        graph::GraphTensor(x),
        graph::GraphTensor(raw_mixes)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4MHCScaleMixes(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->scaled->data(),
        planned->x->data(),
        planned->raw_mixes->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MHCScaleMixes, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_mhc_scale_mixes_impl::infiniop
