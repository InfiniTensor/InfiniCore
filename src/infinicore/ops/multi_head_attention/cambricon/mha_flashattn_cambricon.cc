#if defined(ENABLE_CAMBRICON_API) && defined(ENABLE_FLASH_ATTN)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/cambricon_flash_attention.hpp"
#include "infinicore/ops/mha.hpp"

namespace infinicore::op::mha_impl::cambricon {

struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes))
            : std::nullopt,
        scale,
        is_causal};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    torch_mlu::mlu::MLUStreamGuard guard(
        infinicore::adaptor::get_mlu_stream());

    auto q = infinicore::adaptor::to_aten_tensor(p->q).contiguous();
    auto k = infinicore::adaptor::to_aten_tensor(p->k).contiguous();
    auto v = infinicore::adaptor::to_aten_tensor(p->v).contiguous();
    const bool copy_back = !p->out->is_contiguous();
    Tensor out_work_ic
        = copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor
        = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto out = std::optional<at::Tensor>(out_tensor);
    auto alibi = p->alibi_slopes
                   ? std::optional<at::Tensor>(
                       infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                   : std::nullopt;

    ::mha_fwd(
        q, k, v, out, alibi, 0.0F, p->scale, p->is_causal,
        -1, -1, false, std::nullopt);

    if (copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::cambricon

#endif
