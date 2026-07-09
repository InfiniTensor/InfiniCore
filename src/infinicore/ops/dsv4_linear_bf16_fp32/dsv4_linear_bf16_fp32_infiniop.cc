#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_linear_bf16_fp32.hpp"
#include "infiniop/ops/dsv4_linear_bf16_fp32.h"

namespace infinicore::op::dsv4_linear_bf16_fp32_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4LinearBf16Fp32, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x, w;
};

void *plan(Tensor y, const Tensor &x, const Tensor &w) {
    size_t seed = hash_combine(y, x, w);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dsv4LinearBf16Fp32, seed, y->desc(), x->desc(), w->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4LinearBf16Fp32, descriptor);
    return new PlannedMeta{
        descriptor, graph::GraphTensor(workspace), graph::GraphTensor(y), graph::GraphTensor(x), graph::GraphTensor(w)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4LinearBf16Fp32(
        p->descriptor->desc,
        p->workspace->data(),
        p->workspace->numel(),
        p->y->data(),
        p->x->data(),
        p->w->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4LinearBf16Fp32, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_linear_bf16_fp32_impl::infiniop
