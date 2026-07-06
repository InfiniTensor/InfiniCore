#include "infinicore/ops/unweighted_rms_norm.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::unweighted_rms_norm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, UnweightedRMSNorm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x;
};

void *plan(Tensor y, const Tensor &x, float epsilon) {
    size_t seed = hash_combine(y, x, epsilon);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, UnweightedRMSNorm,
        seed, y->desc(),
        x->desc(),
        epsilon);

    INFINIOP_WORKSPACE_TENSOR(workspace, UnweightedRMSNorm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopUnweightedRMSNorm(
            planned->descriptor->desc,
            planned->workspace->data(),
            planned->workspace->numel(),
            planned->y->data(),
            planned->x->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(UnweightedRMSNorm, &plan, &run, &cleanup);

} // namespace infinicore::op::unweighted_rms_norm_impl::infiniop
