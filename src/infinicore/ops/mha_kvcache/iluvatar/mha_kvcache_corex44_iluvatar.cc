#if defined(ENABLE_ILUVATAR_FLASH_ATTN)
#include "infinicore/ops/mha_kvcache.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ATen.h>

#include <optional>
#include <stdexcept>

namespace ixformer::infer {
void vllm_paged_attention(
    at::Tensor &output,
    at::Tensor &query,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    long num_kv_heads,
    double scale,
    at::Tensor &block_tables,
    at::Tensor &context_lens,
    long block_size,
    long max_context_len,
    const std::optional<at::Tensor> &alibi_slopes,
    bool causal,
    int window_left,
    int window_right,
    int global_window_size,
    double softcap,
    bool use_cuda_graph,
    bool use_sqrt_alibi,
    const std::optional<at::Tensor> &sinks);
}

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
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor = infinicore::adaptor::to_aten_tensor(out_work);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto seqlens_k = infinicore::adaptor::to_aten_tensor(p->seqlens_k);
    auto block_table = infinicore::adaptor::to_aten_tensor(p->block_table);
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

    if (alibi_slopes.has_value()) {
        throw std::runtime_error("Iluvatar vLLM PageAttention does not support alibi_slopes");
    }
    if (q.dim() != 4 || q.size(1) != 1) {
        throw std::runtime_error("Iluvatar vLLM PageAttention expects q with shape [batch, 1, heads, head_size]");
    }
    if (k_cache.dim() != 4 || k_cache.size(1) != v_cache.size(1) ||
        k_cache.size(2) != v_cache.size(2) || k_cache.size(3) != v_cache.size(3)) {
        throw std::runtime_error("Iluvatar vLLM PageAttention expects BSHD k/v caches");
    }

    auto q3 = q.squeeze(1);
    auto out3 = out_tensor.squeeze(1);
    auto k_bhbh = k_cache.permute({0, 2, 1, 3});
    auto v_bhbh = v_cache.permute({0, 2, 1, 3});
    const long kv_num_heads = static_cast<long>(k_cache.size(2));
    const long block_size = static_cast<long>(k_cache.size(1));
    const long max_context_len = block_size * static_cast<long>(block_table.size(1));
    if (q3.size(1) % kv_num_heads != 0) {
        throw std::runtime_error("Iluvatar vLLM PageAttention requires divisible Q/KV heads");
    }

    infinicore::adaptor::set_aten_stream_to_infinicore();
    ixformer::infer::vllm_paged_attention(
        out3,
        q3,
        k_bhbh,
        v_bhbh,
        kv_num_heads,
        p->scale,
        block_table,
        seqlens_k,
        block_size,
        max_context_len,
        std::nullopt,
        true,
        -1,
        -1,
        -1,
        0.0,
        false,
        true,
        std::nullopt);

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
