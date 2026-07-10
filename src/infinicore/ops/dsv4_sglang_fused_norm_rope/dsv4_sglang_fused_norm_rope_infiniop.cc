#include "infinicore/ops/dsv4_sglang_fused_norm_rope.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_fused_norm_rope.h"

namespace infinicore::op::dsv4_sglang_fused_norm_rope_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangFusedNormRope, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, kv, weight, positions, freqs;
};

void *plan(Tensor kv, const Tensor &weight, const Tensor &positions, const Tensor &freqs, double eps) {
    size_t seed = hash_combine(kv, weight, positions, freqs, eps);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangFusedNormRope,
        seed,
        kv->desc(), weight->desc(), positions->desc(), freqs->desc(), eps);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangFusedNormRope, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(kv), graph::GraphTensor(weight), graph::GraphTensor(positions), graph::GraphTensor(freqs)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangFusedNormRope(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->kv->data(), planned->weight->data(), planned->positions->data(), planned->freqs->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangFusedNormRope, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_fused_norm_rope_impl::infiniop
