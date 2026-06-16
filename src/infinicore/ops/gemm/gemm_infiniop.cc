#include "../infiniop_impl.hpp"
#include "infinicore/ops/gemm.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"
#endif

#include <optional>

namespace infinicore::op::gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Gemm, 100);

#ifdef ENABLE_INFINIOPS_API
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
#endif

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a, b;
    float alpha, beta;
#ifdef ENABLE_INFINIOPS_API
    bool use_infiniops = false;
    std::optional<TensorMeta> c_meta, a_meta, b_meta;
#endif
};

void *plan(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
#ifdef ENABLE_INFINIOPS_API
    if (c->device().getType() == Device::Type::NVIDIA) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
        return new PlannedMeta{
            nullptr,
            graph::GraphTensor(c),
            graph::GraphTensor(c),
            graph::GraphTensor(a),
            graph::GraphTensor(b),
            alpha,
            beta,
            true,
            TensorMeta(c),
            TensorMeta(a),
            TensorMeta(b)};
    }
#endif

    size_t seed = hash_combine(c, a, b);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Gemm,
        seed, c->desc(), a->desc(), b->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Gemm, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        alpha, beta};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

#ifdef ENABLE_INFINIOPS_API
    if (planned->use_infiniops) {
        infini::ops::Handle handle;
        handle.set_stream(context::getStream());
        infini::ops::Config config;

        infini::ops::Operator<infini::ops::Gemm>::Call(
            handle,
            config,
            planned->a_meta->tensor(planned->a),
            planned->b_meta->tensor(planned->b),
            std::optional<float>{planned->alpha},
            std::optional<float>{planned->beta},
            std::optional<int>{},
            std::optional<int>{},
            planned->c_meta->tensor(planned->c));
        return;
    }
#endif

    INFINICORE_CHECK_ERROR(infiniopGemm(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->c->data(), planned->a->data(), planned->b->data(), planned->alpha, planned->beta, context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Gemm, &plan, &run, &cleanup);

} // namespace infinicore::op::gemm_impl::infiniop
