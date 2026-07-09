#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_sglang_compress_c128_online_v2_prefill.hpp"
#include "infiniop/ops/dsv4_sglang_compress_c128_online_v2_prefill.h"
namespace infinicore::op::dsv4_sglang_compress_c128_online_v2_prefill_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangCompressC128OnlineV2Prefill, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, kv_score_buffer, kv_score_input, kv_output, ape, plan_c, plan_w;
};
void *plan(const Tensor &kv_score_buffer, const Tensor &kv_score_input, Tensor kv_output, const Tensor &ape, const Tensor &plan_c, const Tensor &plan_w) {
    size_t seed = hash_combine(kv_score_buffer, kv_score_input, kv_output, ape, plan_c, plan_w);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangCompressC128OnlineV2Prefill,
        seed,
        kv_score_buffer->desc(), kv_score_input->desc(), kv_output->desc(), ape->desc(), plan_c->desc(), plan_w->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangCompressC128OnlineV2Prefill, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(kv_score_buffer),
        graph::GraphTensor(kv_score_input),
        graph::GraphTensor(kv_output),
        graph::GraphTensor(ape),
        graph::GraphTensor(plan_c),
        graph::GraphTensor(plan_w)};
}
void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4SglangCompressC128OnlineV2Prefill(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->kv_score_buffer->data(), planned->kv_score_input->data(), planned->kv_output->data(), planned->ape->data(), planned->plan_c->data(), planned->plan_w->data(),
        context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangCompressC128OnlineV2Prefill, &plan, &run, &cleanup);
} // namespace infinicore::op::dsv4_sglang_compress_c128_online_v2_prefill_impl::infiniop
