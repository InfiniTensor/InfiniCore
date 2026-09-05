#include "../infiniop_impl.hpp"
#include "infinicore/ops/grouped_gemm.hpp"

namespace infinicore::op::grouped_gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, GroupedGemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a, b, group_sizes;
    float alpha, beta;
    // Optional host-side group sizes; nullptr falls back to the device sync.
    // Only valid for immediate (non-recorded) execution -- the pointer is not
    // owned, so graph replay must pass nullptr.
    const int32_t *group_sizes_host;
};

void *plan(Tensor c, const Tensor &a, const Tensor &b, const Tensor &group_sizes, float alpha, float beta, const int32_t *group_sizes_host) {
    size_t seed = hash_combine(c, a, b, group_sizes);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, GroupedGemm,
        seed, c->desc(), a->desc(), b->desc(), group_sizes->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, GroupedGemm, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(group_sizes),
        alpha, beta,
        group_sizes_host};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopGroupedGemm(
        planned->descriptor->desc,
        planned->workspace->data(), planned->workspace->numel(),
        planned->c->data(),
        planned->a->data(),
        planned->b->data(),
        planned->group_sizes->data(),
        planned->group_sizes_host,
        planned->alpha,
        planned->beta,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(GroupedGemm, &plan, &run, &cleanup);

} // namespace infinicore::op::grouped_gemm_impl::infiniop
