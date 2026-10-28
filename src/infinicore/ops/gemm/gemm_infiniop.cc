#include "../infiniop_impl.hpp"
#include "infinicore/ops/gemm.hpp"
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_linear_adaptor.hpp"
#endif

namespace infinicore::op::gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Gemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a, b;
    float alpha, beta;
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    at::Tensor vendor_a, vendor_weight, vendor_c;
    bool use_vendor = false;
    bool use_gemv = false;
#endif
};

void *plan(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
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

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (c->device().getType() == Device::Type::ILUVATAR
        && c->dtype() == DataType::BF16 && a->dtype() == c->dtype() && b->dtype() == c->dtype()
        && alpha == 1.0f && beta == 0.0f && a->ndim() == 2 && b->ndim() == 2 && c->ndim() == 2
        && a->is_contiguous() && c->is_contiguous()
        && b->strides()[0] == 1 && b->strides()[1] == b->size(0)) {
        const auto m = a->size(0), k = a->size(1), n = b->size(1);
        const bool gemv = m == 1 && k % 32 == 0 && n % 4 == 0;
        const bool gemm = m >= 2 && m <= 64
                       && (n == k || k == 3 * n || (n >= 2 * k && (n <= 65536 || m >= 32)));
        if ((gemv || gemm) && adaptor::iluvatar_linear::available()) {
            planned->vendor_a = adaptor::to_aten_tensor(planned->a);
            planned->vendor_weight = adaptor::to_aten_tensor(planned->b).transpose(0, 1);
            planned->vendor_c = adaptor::to_aten_tensor(planned->c);
            planned->use_vendor = true;
            planned->use_gemv = gemv;
        }
    }
#endif
    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (planned->use_vendor) {
        adaptor::iluvatar_linear::run(planned->vendor_a, planned->vendor_weight, planned->vendor_c, planned->use_gemv);
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
