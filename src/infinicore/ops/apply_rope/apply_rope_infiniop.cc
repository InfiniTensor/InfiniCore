#include "../infiniop_impl.hpp"
#include "infinicore/ops/apply_rope.hpp"
namespace infinicore::op::apply_rope_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, ApplyRoPE, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, q, k, cos, sin;
};
void *plan(Tensor q, Tensor k, const Tensor &cos, const Tensor &sin) {
    const size_t seed = hash_combine(q, k, cos, sin);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, ApplyRoPE,
                                               seed, q->desc(), k->desc(), cos->desc(), sin->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, ApplyRoPE, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace),
                           graph::GraphTensor(q), graph::GraphTensor(k),
                           graph::GraphTensor(cos), graph::GraphTensor(sin)};
}
void run(void *meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(meta);
    INFINICORE_CHECK_ERROR(infiniopApplyRoPE(p->descriptor->desc,
                                             p->workspace->data(), p->workspace->numel(), p->q->data(), p->k->data(),
                                             p->cos->data(), p->sin->data(), context::getStream()));
}
void cleanup(void **meta) {
    delete *reinterpret_cast<PlannedMeta **>(meta);
    *meta = nullptr;
}
static bool registered = []() {
    ApplyRoPE::plan_dispatcher().registerDevice(Device::Type::ASCEND, &plan);
    ApplyRoPE::run_dispatcher().registerDevice(Device::Type::ASCEND, &run);
    ApplyRoPE::cleanup_dispatcher().registerDevice(Device::Type::ASCEND, &cleanup);
    return true;
}();
} // namespace infinicore::op::apply_rope_impl::infiniop
