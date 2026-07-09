#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_sglang_mega_moe_pre_dispatch.hpp"
#include "infiniop/ops/dsv4_sglang_mega_moe_pre_dispatch.h"
namespace infinicore::op::dsv4_sglang_mega_moe_pre_dispatch_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangMegaMoePreDispatch, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights;
};
void *plan(const Tensor &x, const Tensor &topk_idx, const Tensor &topk_weights, Tensor buf_x, Tensor buf_x_sf, Tensor buf_topk_idx, Tensor buf_topk_weights) {
    size_t seed = hash_combine(x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangMegaMoePreDispatch,
        seed,
        x->desc(), topk_idx->desc(), topk_weights->desc(), buf_x->desc(), buf_x_sf->desc(), buf_topk_idx->desc(), buf_topk_weights->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangMegaMoePreDispatch, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x), graph::GraphTensor(topk_idx), graph::GraphTensor(topk_weights), graph::GraphTensor(buf_x), graph::GraphTensor(buf_x_sf), graph::GraphTensor(buf_topk_idx), graph::GraphTensor(buf_topk_weights)};
}
void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4SglangMegaMoePreDispatch(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x->data(), planned->topk_idx->data(), planned->topk_weights->data(), planned->buf_x->data(), planned->buf_x_sf->data(), planned->buf_topk_idx->data(), planned->buf_topk_weights->data(),
        context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangMegaMoePreDispatch, &plan, &run, &cleanup);
} // namespace infinicore::op::dsv4_sglang_mega_moe_pre_dispatch_impl::infiniop
