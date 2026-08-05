#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::dsa_impl::iluvatar {

void fused_indexer_postprocess(
    Tensor q_out,
    Tensor k_out,
    Tensor weights_out,
    Tensor kv_cache,
    const Tensor &slot_mapping,
    const Tensor &q,
    const Tensor &kw,
    const Tensor &norm_weight,
    const Tensor &norm_bias,
    const Tensor &positions,
    const Tensor &cos_sin_cache,
    int64_t num_cache_tokens,
    bool is_neox,
    double eps,
    double weights_scale) {
    if (!adaptor::iluvatar_vendor::fused_deepseek_v2_indexer_postprocess_available()) {
        throw std::runtime_error("fused_deepseek_v2_indexer_postprocess requires the Iluvatar vendor extension");
    }
    auto q_out_at = adaptor::to_aten_tensor(q_out);
    auto k_out_at = adaptor::to_aten_tensor(k_out);
    auto weights_out_at = adaptor::to_aten_tensor(weights_out);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
    auto q_at = adaptor::to_aten_tensor(q);
    auto kw_at = adaptor::to_aten_tensor(kw);
    auto norm_weight_at = adaptor::to_aten_tensor(norm_weight);
    auto norm_bias_at = adaptor::to_aten_tensor(norm_bias);
    auto positions_at = adaptor::to_aten_tensor(positions);
    auto cos_sin_cache_at = adaptor::to_aten_tensor(cos_sin_cache);
    adaptor::iluvatar_vendor::fused_deepseek_v2_indexer_postprocess(
        q_out_at, k_out_at,
        weights_out_at, kv_cache_at,
        slot_mapping_at, q_at,
        kw_at, norm_weight_at,
        norm_bias_at, positions_at,
        cos_sin_cache_at, num_cache_tokens, is_neox, eps,
        weights_scale);
}

void indexer_k_cache(const Tensor &k, Tensor kv_cache, const Tensor &slot_mapping) {
    if (!adaptor::iluvatar_vendor::indexer_k_cache_available()) {
        throw std::runtime_error("indexer_k_cache requires the Iluvatar vendor extension");
    }
    auto k_at = adaptor::to_aten_tensor(k);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
    adaptor::iluvatar_vendor::indexer_k_cache(
        k_at, kv_cache_at,
        slot_mapping_at);
}

void indexer_k_quant_and_cache(const Tensor &k,
                               Tensor kv_cache,
                               const Tensor &slot_mapping,
                               int64_t quant_block_size,
                               const std::string &scale_fmt) {
    if (!adaptor::iluvatar_vendor::indexer_k_quant_and_cache_available()) {
        throw std::runtime_error("indexer_k_quant_and_cache requires the Iluvatar vendor extension");
    }
    auto k_at = adaptor::to_aten_tensor(k);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
    adaptor::iluvatar_vendor::indexer_k_quant_and_cache(
        k_at, kv_cache_at,
        slot_mapping_at, quant_block_size, scale_fmt);
}

void block_sparse_mqa_logits(Tensor logits,
                             const Tensor &q,
                             const Tensor &kv_cache,
                             const Tensor &cu_seqlens_q,
                             const Tensor &cu_seqlens_kv,
                             const Tensor &block_table,
                             const Tensor &weights,
                             int64_t max_q_len,
                             int64_t max_kv_len,
                             int64_t max_context_len) {
    if (!adaptor::iluvatar_vendor::compute_block_sparse_mqa_logits_available()) {
        throw std::runtime_error("compute_block_sparse_mqa_logits requires the Iluvatar vendor extension");
    }
    auto q_at = adaptor::to_aten_tensor(q);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto cu_seqlens_q_at = adaptor::to_aten_tensor(cu_seqlens_q);
    auto cu_seqlens_kv_at = adaptor::to_aten_tensor(cu_seqlens_kv);
    auto block_table_at = adaptor::to_aten_tensor(block_table);
    auto weights_at = adaptor::to_aten_tensor(weights);
    auto logits_at = adaptor::to_aten_tensor(logits);
    adaptor::iluvatar_vendor::compute_block_sparse_mqa_logits(
        q_at, kv_cache_at,
        cu_seqlens_q_at, cu_seqlens_kv_at,
        block_table_at, weights_at,
        logits_at, max_q_len, max_kv_len, max_context_len);
}

void select_prefill_topk(Tensor topk_indices,
                         const Tensor &logits,
                         const Tensor &cu_seqlen_ks,
                         const Tensor &cu_seqlen_ke) {
    auto logits_at = adaptor::to_aten_tensor(logits);
    auto cu_seqlen_ks_at = adaptor::to_aten_tensor(cu_seqlen_ks);
    auto cu_seqlen_ke_at = adaptor::to_aten_tensor(cu_seqlen_ke);
    auto topk_indices_at = adaptor::to_aten_tensor(topk_indices);
    adaptor::iluvatar_vendor::select_prefill_topk_block_indices(
        logits_at, cu_seqlen_ks_at,
        cu_seqlen_ke_at, topk_indices_at);
}

void select_decode_topk(Tensor topk_indices,
                        const Tensor &logits,
                        const Tensor &seq_lens) {
    auto logits_at = adaptor::to_aten_tensor(logits);
    auto seq_lens_at = adaptor::to_aten_tensor(seq_lens);
    auto topk_indices_at = adaptor::to_aten_tensor(topk_indices);
    adaptor::iluvatar_vendor::select_decode_topk_block_indices(
        logits_at, seq_lens_at,
        topk_indices_at);
}

void map_prefill_indices(Tensor output,
                         const Tensor &req_id,
                         const Tensor &block_table,
                         const Tensor &token_indices,
                         int64_t block_size,
                         bool has_prefill_workspace,
                         std::optional<Tensor> prefill_workspace_request_ids,
                         std::optional<Tensor> prefill_workspace_starts) {
    std::optional<at::Tensor> request_ids_at;
    std::optional<at::Tensor> starts_at;
    if (prefill_workspace_request_ids) {
        request_ids_at = adaptor::to_aten_tensor(*prefill_workspace_request_ids);
    }
    if (prefill_workspace_starts) {
        starts_at = adaptor::to_aten_tensor(*prefill_workspace_starts);
    }
    auto output_at = adaptor::to_aten_tensor(output);
    auto req_id_at = adaptor::to_aten_tensor(req_id);
    auto block_table_at = adaptor::to_aten_tensor(block_table);
    auto token_indices_at = adaptor::to_aten_tensor(token_indices);
    adaptor::iluvatar_vendor::map_prefill_request_block_indices(
        output_at, req_id_at,
        block_table_at, token_indices_at,
        block_size, has_prefill_workspace, request_ids_at, starts_at);
}

void map_decode_indices(Tensor output,
                        const Tensor &req_id,
                        const Tensor &block_table,
                        const Tensor &token_indices,
                        int64_t block_size) {
    auto output_at = adaptor::to_aten_tensor(output);
    auto req_id_at = adaptor::to_aten_tensor(req_id);
    auto block_table_at = adaptor::to_aten_tensor(block_table);
    auto token_indices_at = adaptor::to_aten_tensor(token_indices);
    adaptor::iluvatar_vendor::map_decode_request_block_indices(
        output_at, req_id_at,
        block_table_at, token_indices_at,
        block_size);
}

void topk_context_lens(Tensor topk_lens, const Tensor &indices) {
    auto topk_lens_at = adaptor::to_aten_tensor(topk_lens);
    auto indices_at = adaptor::to_aten_tensor(indices);
    adaptor::iluvatar_vendor::topk_indices_context_lens(
        topk_lens_at, indices_at);
}

void sparse_flash_mla(Tensor output,
                      const Tensor &query,
                      const Tensor &kv_cache,
                      const Tensor &indices,
                      const Tensor &topk_lens,
                      float scale,
                      std::optional<Tensor> attn_sink) {
    if (!adaptor::iluvatar_vendor::sparse_flash_mla_available()) {
        throw std::runtime_error("sparse_flash_mla requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> sink_at;
    if (attn_sink) {
        sink_at = adaptor::to_aten_tensor(*attn_sink);
    }
    auto output_at = adaptor::to_aten_tensor(output);
    auto query_at = adaptor::to_aten_tensor(query);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto indices_at = adaptor::to_aten_tensor(indices);
    auto topk_lens_at = adaptor::to_aten_tensor(topk_lens);
    adaptor::iluvatar_vendor::sparse_flash_mla(
        output_at, query_at,
        kv_cache_at, indices_at,
        topk_lens_at, scale, sink_at);
}

static bool registered = []() {
    vendor_ops::fused_indexer_postprocess_dispatcher().registerDevice(Device::Type::ILUVATAR, &fused_indexer_postprocess);
    vendor_ops::indexer_k_cache_dispatcher().registerDevice(Device::Type::ILUVATAR, &indexer_k_cache);
    vendor_ops::indexer_k_quant_and_cache_dispatcher().registerDevice(Device::Type::ILUVATAR, &indexer_k_quant_and_cache);
    vendor_ops::block_sparse_mqa_logits_dispatcher().registerDevice(Device::Type::ILUVATAR, &block_sparse_mqa_logits);
    vendor_ops::select_prefill_topk_dispatcher().registerDevice(Device::Type::ILUVATAR, &select_prefill_topk);
    vendor_ops::select_decode_topk_dispatcher().registerDevice(Device::Type::ILUVATAR, &select_decode_topk);
    vendor_ops::map_prefill_indices_dispatcher().registerDevice(Device::Type::ILUVATAR, &map_prefill_indices);
    vendor_ops::map_decode_indices_dispatcher().registerDevice(Device::Type::ILUVATAR, &map_decode_indices);
    vendor_ops::topk_context_lens_dispatcher().registerDevice(Device::Type::ILUVATAR, &topk_context_lens);
    vendor_ops::sparse_flash_mla_dispatcher().registerDevice(Device::Type::ILUVATAR, &sparse_flash_mla);
    return true;
}();

} // namespace infinicore::op::dsa_impl::iluvatar
#endif
