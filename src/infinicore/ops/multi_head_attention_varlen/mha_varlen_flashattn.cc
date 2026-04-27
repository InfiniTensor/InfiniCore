#include "infinicore/ops/mha_varlen.hpp"

#include "infinicore/adaptor/flash_attention_adaptor.hpp"

#include <stdexcept>

#ifdef ENABLE_FLASH_ATTN
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

namespace infinicore::op::mha_varlen_impl::flashattn {

struct PlannedMeta {
    graph::GraphTensor out, q, k, v, cum_seqlens_q, cum_seqlens_k, block_table;
    int max_seqlen_q, max_seqlen_k;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &cum_seqlens_q,
           const Tensor &cum_seqlens_k,
           const Tensor &block_table,
           int max_seqlen_q,
           int max_seqlen_k,
           std::optional<Tensor> alibi_slopes,
           float scale) {

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(cum_seqlens_q),
        graph::GraphTensor(cum_seqlens_k),
        graph::GraphTensor(block_table),
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

namespace {

#ifdef ENABLE_FLASH_ATTN
// MetaX/hpcc pip `flash_attn_2_cuda` exports `mha_varlen_fwd` at global scope (no namespace),
// while NVIDIA `flash-attn-nvidia.so` uses `flash::mha_varlen_fwd`.
#if defined(ENABLE_METAX_API)
#define INFINICORE_FLASH_OP(name) ::name
#else
#define INFINICORE_FLASH_OP(name) flash::name
#endif

#endif // ENABLE_FLASH_ATTN
} // namespace

void run(void *planned_meta) {
#ifdef ENABLE_FLASH_ATTN
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto out = std::optional<at::Tensor>(out_work);

    auto cu_seqlens_q = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_q);
    auto cu_seqlens_kv = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_k);
    std::optional<at::Tensor> seqused_k = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    auto block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));
    auto max_seqlen_q = p->max_seqlen_q;
    auto max_seqlen_k = p->max_seqlen_k;
    auto alibi_slopes =
        p->alibi_slopes ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes)) : std::nullopt;
    auto scale = p->scale;

#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
    std::optional<at::Tensor> flash_attn_mars_ext = std::nullopt;
#endif

    INFINICORE_FLASH_OP(mha_varlen_fwd)(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        scale,
        false,
        true,
        -1,
        -1,
        0.0,
        false,
        std::nullopt
#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
        ,
        flash_attn_mars_ext
#endif
    );

    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }

#else
    throw std::runtime_error("FlashAttention is not enabled in this build");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MultiheadAttentionVarlen, &plan, &run, &cleanup);

} // namespace infinicore::op::mha_varlen_impl::flashattn
