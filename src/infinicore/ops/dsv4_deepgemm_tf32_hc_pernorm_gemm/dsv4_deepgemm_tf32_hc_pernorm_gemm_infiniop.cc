#include "infinicore/ops/dsv4_deepgemm_tf32_hc_pernorm_gemm.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/dsv4_deepgemm_tf32_hc_pernorm_gemm.h"

namespace infinicore::op::dsv4_deepgemm_tf32_hc_pernorm_gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4DeepgemmTf32HcPernormGemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, a, b, d, sqr_sum;
};

void *plan(const Tensor &a, const Tensor &b, Tensor d, Tensor sqr_sum, int64_t num_splits) {
    size_t seed = hash_combine(a, b, d, sqr_sum, num_splits);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4DeepgemmTf32HcPernormGemm,
        seed,
        a->desc(), b->desc(), d->desc(), sqr_sum->desc(), num_splits);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4DeepgemmTf32HcPernormGemm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(d),
        graph::GraphTensor(sqr_sum)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDsv4DeepgemmTf32HcPernormGemm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->a->data(),
        planned->b->data(),
        planned->d->data(),
        planned->sqr_sum->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4DeepgemmTf32HcPernormGemm, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_deepgemm_tf32_hc_pernorm_gemm_impl::infiniop
