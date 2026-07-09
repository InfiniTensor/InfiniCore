#include "infinicore/ops/dsv4_sglang_fused_rope.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_sglang_fused_rope.h"

namespace infinicore::op::dsv4_sglang_fused_rope_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4SglangFusedRope, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, q, freqs_cis, positions;
};

void *plan(Tensor q, const Tensor &freqs_cis, const Tensor &positions, bool inverse) {
    size_t seed = hash_combine(q, freqs_cis, positions, inverse);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4SglangFusedRope,
        seed,
        q->desc(), freqs_cis->desc(), positions->desc(), inverse);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4SglangFusedRope, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(q),
        graph::GraphTensor(freqs_cis),
        graph::GraphTensor(positions)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4SglangFusedRope(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->q->data(),
        planned->freqs_cis->data(),
        planned->positions->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4SglangFusedRope, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_sglang_fused_rope_impl::infiniop
