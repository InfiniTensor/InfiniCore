#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_fused_rope.hpp"
#include "infiniop/ops/dsv4_fused_rope.h"

#include <optional>

namespace infinicore::op::dsv4_fused_rope_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4FusedRope, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, q, freq_real, freq_imag;
    std::optional<graph::GraphTensor> k;
    bool has_k;
};

void *plan(Tensor q, Tensor k, const Tensor &freq_real, const Tensor &freq_imag, bool has_k) {
    size_t seed = has_k ? hash_combine(q, k, freq_real, freq_imag, 1) : hash_combine(q, freq_real, freq_imag, 0);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4FusedRope,
        seed,
        q->desc(),
        has_k ? k->desc() : nullptr,
        freq_real->desc(),
        freq_imag->desc(),
        has_k ? 1 : 0);
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4FusedRope, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(q),
        graph::GraphTensor(freq_real),
        graph::GraphTensor(freq_imag),
        has_k ? std::optional<graph::GraphTensor>(graph::GraphTensor(k)) : std::nullopt,
        has_k};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4FusedRope(
        p->descriptor->desc,
        p->workspace->data(),
        p->workspace->numel(),
        p->q->data(),
        p->has_k ? p->k.value()->data() : nullptr,
        p->freq_real->data(),
        p->freq_imag->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4FusedRope, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_fused_rope_impl::infiniop
