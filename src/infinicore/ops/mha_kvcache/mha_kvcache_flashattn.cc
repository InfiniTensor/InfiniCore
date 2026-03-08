#include "infinicore/ops/mha_kvcache.hpp"

#include "infinicore/adaptor/flash_attention_adaptor.hpp"

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
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto out       = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->out));
    auto q         = infinicore::adaptor::to_aten_tensor(p->q);
    auto k_cache   = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache   = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto seqlens_k = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(p->seqlens_k));
    auto block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));
    auto alibi_slopes = p->alibi_slopes
        ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
        : std::nullopt;

    // No new KV tokens to append (pure decode, KV already written to cache).
    std::optional<const at::Tensor> k_new        = std::nullopt;
    std::optional<const at::Tensor> v_new        = std::nullopt;
    std::optional<const at::Tensor> rotary_cos   = std::nullopt;
    std::optional<const at::Tensor> rotary_sin   = std::nullopt;
    std::optional<const at::Tensor> cache_batch_idx = std::nullopt;
    std::optional<const at::Tensor> leftpad_k    = std::nullopt;

    flash::mha_fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out,
        p->scale,
        true,   // is_causal
        -1,     // window_size_left  (-1 = no sliding window)
        -1,     // window_size_right
        0.0f,   // softcap
        false,  // is_rotary_interleaved
        0       // num_splits (0 = auto)
    );
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MhaKVCache, &plan, &run, &cleanup);

} // namespace infinicore::op::mha_kvcache_impl::flashattn
