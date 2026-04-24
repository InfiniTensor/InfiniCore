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
struct VarlenFlashPrepared {
    Tensor k_work;
    Tensor v_work;
    Tensor out_work_ic;
    at::Tensor q;
    at::Tensor k;
    at::Tensor v;
    bool out_need_copy_back;
    std::optional<at::Tensor> out_opt;
    at::Tensor cu_seqlens_q;
    at::Tensor cu_seqlens_kv;
    std::optional<at::Tensor> block_table;
    std::optional<at::Tensor> alibi_slopes;
    int max_seqlen_q;
    int max_seqlen_k;
    float scale;
};

VarlenFlashPrepared prepare_varlen_flash_tensors(PlannedMeta *p) {
    VarlenFlashPrepared t;
    // Varlen flash-attn: keep k/v contiguous for dense/paged layout; avoid extra copies for q/metadata when already dense.
    t.q = infinicore::adaptor::to_aten_tensor(p->q);
    t.k_work = p->k->contiguous();
    t.v_work = p->v->contiguous();
    t.k = infinicore::adaptor::to_aten_tensor(t.k_work);
    t.v = infinicore::adaptor::to_aten_tensor(t.v_work);

    t.out_need_copy_back = !p->out->is_contiguous();
    t.out_work_ic = t.out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(t.out_work_ic);
    t.out_opt = std::optional<at::Tensor>(out_work);
    t.cu_seqlens_q = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_q);
    t.cu_seqlens_kv = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_k);
    t.block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));
    t.max_seqlen_q = p->max_seqlen_q;
    t.max_seqlen_k = p->max_seqlen_k;
    t.alibi_slopes = p->alibi_slopes
                       ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                       : std::nullopt;
    t.scale = p->scale;
    return t;
}

void copy_varlen_flash_output_back(PlannedMeta *p, VarlenFlashPrepared &t) {
    if (t.out_need_copy_back) {
        p->out->copy_from(t.out_work_ic);
    }
}

#if defined(ENABLE_METAX_API)
/**
 * MetaX / hpcc: pip `flash_attn_2_cuda` exposes `mha_varlen_fwd` in the global namespace and
 * relies on the current CUDA stream. Kept separate from the NVIDIA `run()` body to reduce merge conflicts.
 */
void run_flashattn_varlen_metax(PlannedMeta *p) {
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto t = prepare_varlen_flash_tensors(p);
    std::optional<at::Tensor> seqused_k = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    // `flash_attn_2_cuda` (pip / MetaX) may append an extra trailing argument (`flash_attn_mars_ext_`)
    // depending on the HPCC/MetaX stack version.
#if defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
    std::optional<at::Tensor> flash_attn_mars_ext = std::nullopt;
#endif
    ::mha_varlen_fwd(
        t.q,
        t.k,
        t.v,
        t.out_opt,
        t.cu_seqlens_q,
        t.cu_seqlens_kv,
        seqused_k,
        leftpad_k,
        t.block_table,
        t.alibi_slopes,
        t.max_seqlen_q,
        t.max_seqlen_k,
        0.0,
        t.scale,
        false,
        true,
        -1,
        -1,
        0.0,
        false,
        std::nullopt
#if defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
        ,
        flash_attn_mars_ext
#endif
    );
    copy_varlen_flash_output_back(p, t);
}
#endif

#endif // ENABLE_FLASH_ATTN
} // namespace

void run(void *planned_meta) {
#ifdef ENABLE_FLASH_ATTN
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

#if defined(ENABLE_METAX_API)
    run_flashattn_varlen_metax(p);
    return;
#endif

    // Original InfiniCore path (NVIDIA + xmake flash-attn-nvidia). MetaX is handled above.
#if defined(ENABLE_NVIDIA_API)
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);
    auto out = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->out));
    auto cu_seqlens_q = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_q);
    auto cu_seqlens_kv = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_k);
    std::optional<at::Tensor> seqused_k = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    auto block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));
    auto max_seqlen_q = p->max_seqlen_q;
    auto max_seqlen_k = p->max_seqlen_k;
    auto alibi_slopes = p->alibi_slopes ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes)) : std::nullopt;
    auto scale = p->scale;

    flash::mha_varlen_fwd(
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
        std::nullopt);
#else
    throw std::runtime_error("FlashAttention varlen: no supported GPU backend in this build");
#endif

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
