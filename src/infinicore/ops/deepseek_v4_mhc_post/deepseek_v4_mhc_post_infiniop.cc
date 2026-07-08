#include "infinicore/ops/deepseek_v4_mhc_post.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::deepseek_v4_mhc_post_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekV4MHCPost, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor y;
    graph::GraphTensor new_x;
    graph::GraphTensor residual;
    graph::GraphTensor post;
    graph::GraphTensor comb;
};

void *plan(Tensor y,
           const Tensor &new_x,
           const Tensor &residual,
           const Tensor &post,
           const Tensor &comb) {
    size_t seed = hash_combine(y, new_x, residual, post, comb);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        DeepseekV4MHCPost,
        seed,
        y->desc(),
        new_x->desc(),
        residual->desc(),
        post->desc(),
        comb->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekV4MHCPost, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(new_x),
        graph::GraphTensor(residual),
        graph::GraphTensor(post),
        graph::GraphTensor(comb)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDeepseekV4MHCPost(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->y->data(),
        planned->new_x->data(),
        planned->residual->data(),
        planned->post->data(),
        planned->comb->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4MHCPost, &plan, &run, cleanup);

} // namespace infinicore::op::deepseek_v4_mhc_post_impl::infiniop
