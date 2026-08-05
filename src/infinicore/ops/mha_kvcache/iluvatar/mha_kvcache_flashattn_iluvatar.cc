#if defined(ENABLE_ILUVATAR_FLASH_ATTN)
#include "infinicore/ops/mha_kvcache.hpp"

#include "infinicore/adaptor/iluvatar_flash_attention_adaptor.hpp"
#include <ATen/ops/arange.h>

#include <stdexcept>

namespace infinicore::op::mha_kvcache_impl::flashattn {

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &seqlens_k,
           const Tensor &block_table,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(seqlens_k),
        graph::GraphTensor(block_table),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    infinicore::adaptor::set_aten_stream_to_infinicore();
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    // Paged KV caches must be contiguous for flash-attn; avoid extra copies for q/metadata when already dense.
    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor = infinicore::adaptor::to_aten_tensor(out_work);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

    if (alibi_slopes.has_value()) {
        throw std::runtime_error("Iluvatar MHA KVCache does not support alibi_slopes");
    }

    std::optional<const at::Tensor> k_new, v_new, q_v;
    std::optional<const at::Tensor> cu_seqlens_k, cu_seqlens_k_new;
    std::optional<const at::Tensor> seqused_q;
    auto seqused_k = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(p->seqlens_k));
    auto block_table_const = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));
    std::optional<const at::Tensor> kv_batch_idx, leftpad_k;
    std::optional<const at::Tensor> rotary_cos, rotary_sin, seqlens_rotary;
    std::optional<at::Tensor> q_descale, k_descale, v_descale;
    std::optional<at::Tensor> scheduler_metadata;
    std::optional<const at::Tensor> s_aux, cp_tot_seqused_k;

    if (q.dim() != 4 || q.size(1) != 1) {
        throw std::runtime_error("Iluvatar MHA KVCache expects q with shape [batch, 1, heads, head_size]");
    }
    auto q_fwd = q.squeeze(1);
    auto out_fwd = out_tensor.squeeze(1);
    auto cu_seqlens_q_tensor = at::arange(q.size(0) + 1, q.options().dtype(at::kInt));
    auto cu_seqlens_q = std::optional<const at::Tensor>(cu_seqlens_q_tensor);

    const bool use_dynamic_out = q.dim() == 4 && k_cache.dim() == 4
                              && q.size(1) == 1 && q.size(2) > k_cache.size(2)
                              && q.size(3) % 8 == 0;
    auto out = use_dynamic_out ? std::optional<at::Tensor>(std::nullopt)
                               : std::optional<at::Tensor>(out_fwd);
    const int max_seqlen_q = q.dim() == 4 ? static_cast<int>(q.size(1)) : 1;
    const int max_seqlen_k = static_cast<int>(block_table_const->size(1) * k_cache.size(1));

    auto result = pyinfer::cuinfer::mha_fwd(
        q_fwd,
        k_cache,
        v_cache,
        k_new,
        v_new,
        q_v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        block_table_const,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        p->scale,
        true,
        -1,
        -1,
        0.0f,
        true,
        scheduler_metadata,
        0,
        std::nullopt,
        0,
        s_aux,
        1,
        0,
        cp_tot_seqused_k);

    if (use_dynamic_out) {
        out_fwd.copy_(result[0]);
    }
    if (out_need_copy_back) {
        p->out->copy_from(out_work);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MhaKVCache, &plan, &run, &cleanup);

} // namespace infinicore::op::mha_kvcache_impl::flashattn
#endif
