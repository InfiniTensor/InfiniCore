#if defined(ENABLE_ALI_API) && defined(ENABLE_FLASH_ATTN)
#include "infinicore/ops/mha.hpp"

#include "infinicore/adaptor/flash_attention_adaptor.hpp"

#include <c10/cuda/CUDAGuard.h>

namespace infinicore::op::mha_impl::flashattn {

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
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        is_causal};
}

void run(void *planned_meta) {
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto out = std::optional<at::Tensor>(out_work);

    auto alibi_slopes = p->alibi_slopes ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes)) : std::nullopt;
    auto scale = p->scale;
    auto is_causal = p->is_causal;

    flash::mha_fwd(
        q,
        k,
        v,
        out,
        alibi_slopes,
        0.0,
        scale,
        is_causal,
        -1,
        -1,
        0.0,
        false,
        std::nullopt);

    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() { MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::ALI, &plan); MultiheadAttention::run_dispatcher().registerDevice(Device::Type::ALI, &run); MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::ALI, &cleanup); return true; }();

} // namespace infinicore::op::mha_impl::flashattn
#endif
