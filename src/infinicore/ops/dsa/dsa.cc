#include "infinicore/ops/dsa.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/fp8_sparse_mla.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {
namespace {

class DeferredGraphOperator final : public graph::GraphOperator {
public:
    explicit DeferredGraphOperator(std::function<void()> runner)
        : runner_(std::move(runner)) {}

    void run() const override { runner_(); }

private:
    std::function<void()> runner_;
};

bool defer_if_recording(std::function<void()> runner) {
    if (!context::isGraphRecording()) {
        return false;
    }
    context::addGraphOperator(
        std::make_shared<DeferredGraphOperator>(std::move(runner)));
    return true;
}

void require_iluvatar(const Tensor &tensor, const char *op_name) {
    if (!tensor || tensor->device().getType() != Device::Type::ILUVATAR) {
        throw std::runtime_error(std::string(op_name) + " currently supports only Iluvatar tensors");
    }
}

void require_contiguous(const Tensor &tensor, const char *op_name) {
    if (!tensor || !tensor->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors");
    }
}

void require_i32(const Tensor &tensor, const char *name) {
    if (tensor->dtype() != DataType::I32) {
        throw std::runtime_error(std::string(name) + " must be int32");
    }
}

} // namespace

void fused_deepseek_v2_indexer_postprocess_(
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
    require_iluvatar(q_out, "fused_deepseek_v2_indexer_postprocess");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        q_out, k_out, weights_out, kv_cache, slot_mapping, q, kw, norm_weight,
        norm_bias, positions, cos_sin_cache);
    for (const auto &tensor : {q_out, k_out, weights_out, kv_cache, slot_mapping,
                               q, kw, norm_weight, norm_bias, positions,
                               cos_sin_cache}) {
        require_contiguous(tensor, "fused_deepseek_v2_indexer_postprocess");
    }
    if (q->ndim() != 3 || kw->ndim() != 2 || q_out->shape() != q->shape()
        || weights_out->ndim() != 2 || positions->ndim() != 1
        || slot_mapping->ndim() != 1 || cos_sin_cache->ndim() != 2) {
        throw std::runtime_error("fused_deepseek_v2_indexer_postprocess tensor rank/shape mismatch");
    }
    if (q->dtype() != DataType::F16 && q->dtype() != DataType::BF16) {
        throw std::runtime_error("fused_deepseek_v2_indexer_postprocess requires fp16/bfloat16");
    }
    if (positions->dtype() != DataType::I64 || slot_mapping->dtype() != DataType::I64) {
        throw std::runtime_error("fused_deepseek_v2_indexer_postprocess positions and slot_mapping must be int64");
    }
    if (num_cache_tokens < 0 || static_cast<size_t>(num_cache_tokens) > slot_mapping->numel()) {
        throw std::runtime_error("fused_deepseek_v2_indexer_postprocess invalid num_cache_tokens");
    }
    if (defer_if_recording([q_out, k_out, weights_out, kv_cache, slot_mapping, q, kw,
                            norm_weight, norm_bias, positions, cos_sin_cache,
                            num_cache_tokens, is_neox, eps, weights_scale] {
            fused_deepseek_v2_indexer_postprocess_(
                q_out, k_out, weights_out, kv_cache, slot_mapping, q, kw,
                norm_weight, norm_bias, positions, cos_sin_cache,
                num_cache_tokens, is_neox, eps, weights_scale);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (!adaptor::vllm_iluvatar::fused_deepseek_v2_indexer_postprocess_available()) {
        throw std::runtime_error("vllm_iluvatar fused indexer postprocess is unavailable");
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
    adaptor::vllm_iluvatar::fused_deepseek_v2_indexer_postprocess(
        q_out_at, k_out_at, weights_out_at, kv_cache_at, slot_mapping_at, q_at,
        kw_at, norm_weight_at, norm_bias_at, positions_at, cos_sin_cache_at,
        num_cache_tokens, is_neox, eps, weights_scale);
    return;
#else
    throw std::runtime_error("fused_deepseek_v2_indexer_postprocess requires an Iluvatar ATen build");
#endif
}

void indexer_k_cache_(const Tensor &k, Tensor kv_cache, const Tensor &slot_mapping) {
    require_iluvatar(k, "indexer_k_cache");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k, kv_cache, slot_mapping);
    require_contiguous(k, "indexer_k_cache");
    require_contiguous(kv_cache, "indexer_k_cache");
    require_contiguous(slot_mapping, "indexer_k_cache");
    if (k->ndim() != 2 || kv_cache->ndim() != 3 || slot_mapping->ndim() != 1
        || k->size(1) != kv_cache->size(2) || slot_mapping->dtype() != DataType::I64) {
        throw std::runtime_error("indexer_k_cache tensor shape/dtype mismatch");
    }
    if (defer_if_recording([k, kv_cache, slot_mapping] {
            indexer_k_cache_(k, kv_cache, slot_mapping);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (!adaptor::vllm_iluvatar::indexer_k_cache_available()) {
        throw std::runtime_error("vllm_iluvatar indexer_k_cache is unavailable");
    }
    auto k_at = adaptor::to_aten_tensor(k);
    auto cache_at = adaptor::to_aten_tensor(kv_cache);
    auto slots_at = adaptor::to_aten_tensor(slot_mapping);
    adaptor::vllm_iluvatar::indexer_k_cache(k_at, cache_at, slots_at);
    return;
#else
    throw std::runtime_error("indexer_k_cache requires an Iluvatar ATen build");
#endif
}

void indexer_k_quant_and_cache_(
    const Tensor &k,
    Tensor kv_cache,
    const Tensor &slot_mapping,
    int64_t quant_block_size,
    const std::string &scale_fmt) {
    require_iluvatar(k, "indexer_k_quant_and_cache");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k, kv_cache, slot_mapping);
    require_contiguous(k, "indexer_k_quant_and_cache");
    require_contiguous(kv_cache, "indexer_k_quant_and_cache");
    require_contiguous(slot_mapping, "indexer_k_quant_and_cache");
    if (k->ndim() != 2 || kv_cache->ndim() != 3 || slot_mapping->ndim() != 1
        || kv_cache->dtype() != DataType::U8 || slot_mapping->dtype() != DataType::I64
        || k->size(0) != slot_mapping->numel()
        || kv_cache->size(2) != k->size(1) + sizeof(float)
        || quant_block_size != static_cast<int64_t>(k->size(1))) {
        throw std::runtime_error("indexer_k_quant_and_cache tensor shape/dtype mismatch");
    }
    if (scale_fmt != "ue8m0" && !scale_fmt.empty()) {
        throw std::runtime_error("indexer_k_quant_and_cache supports ue8m0 or empty scale_fmt");
    }
    if (defer_if_recording([k, kv_cache, slot_mapping, quant_block_size, scale_fmt] {
            indexer_k_quant_and_cache_(
                k, kv_cache, slot_mapping, quant_block_size, scale_fmt);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (!adaptor::vllm_iluvatar::indexer_k_quant_and_cache_available()) {
        throw std::runtime_error("vllm_iluvatar indexer_k_quant_and_cache is unavailable");
    }
    auto k_at = adaptor::to_aten_tensor(k);
    auto cache_at = adaptor::to_aten_tensor(kv_cache);
    auto slots_at = adaptor::to_aten_tensor(slot_mapping);
    adaptor::vllm_iluvatar::indexer_k_quant_and_cache(
        k_at, cache_at, slots_at, quant_block_size, scale_fmt);
    return;
#else
    throw std::runtime_error(
        "indexer_k_quant_and_cache requires an Iluvatar ATen build");
#endif
}

void compute_block_sparse_mqa_logits_(
    Tensor logits,
    const Tensor &q,
    const Tensor &kv_cache,
    const Tensor &cu_seqlens_q,
    const Tensor &cu_seqlens_kv,
    const Tensor &block_table,
    const Tensor &weights,
    int64_t max_q_len,
    int64_t max_kv_len,
    int64_t max_context_len) {
    require_iluvatar(q, "compute_block_sparse_mqa_logits");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        logits, q, kv_cache, cu_seqlens_q, cu_seqlens_kv, block_table, weights);
    for (const auto &tensor : {logits, q, kv_cache, cu_seqlens_q, cu_seqlens_kv,
                               block_table, weights}) {
        require_contiguous(tensor, "compute_block_sparse_mqa_logits");
    }
    if (q->ndim() != 3 || kv_cache->ndim() != 3 || logits->ndim() != 2
        || weights->ndim() != 2 || block_table->ndim() != 2) {
        throw std::runtime_error("compute_block_sparse_mqa_logits tensor rank mismatch");
    }
    require_i32(cu_seqlens_q, "cu_seqlens_q");
    require_i32(cu_seqlens_kv, "cu_seqlens_kv");
    require_i32(block_table, "block_table");
    if (max_q_len <= 0 || max_kv_len <= 0 || max_context_len <= 0) {
        throw std::runtime_error("compute_block_sparse_mqa_logits lengths must be positive");
    }
    if (defer_if_recording([logits, q, kv_cache, cu_seqlens_q, cu_seqlens_kv, block_table,
                            weights, max_q_len, max_kv_len, max_context_len] {
            compute_block_sparse_mqa_logits_(
                logits, q, kv_cache, cu_seqlens_q, cu_seqlens_kv,
                block_table, weights, max_q_len, max_kv_len, max_context_len);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (!adaptor::vllm_iluvatar::compute_block_sparse_mqa_logits_available()) {
        throw std::runtime_error("vllm_iluvatar block sparse logits is unavailable");
    }
    auto logits_at = adaptor::to_aten_tensor(logits);
    auto q_at = adaptor::to_aten_tensor(q);
    auto cache_at = adaptor::to_aten_tensor(kv_cache);
    auto cu_q_at = adaptor::to_aten_tensor(cu_seqlens_q);
    auto cu_kv_at = adaptor::to_aten_tensor(cu_seqlens_kv);
    auto blocks_at = adaptor::to_aten_tensor(block_table);
    auto weights_at = adaptor::to_aten_tensor(weights);
    adaptor::vllm_iluvatar::compute_block_sparse_mqa_logits(
        q_at, cache_at, cu_q_at, cu_kv_at, blocks_at, weights_at, logits_at,
        max_q_len, max_kv_len, max_context_len);
    return;
#else
    throw std::runtime_error("compute_block_sparse_mqa_logits requires an Iluvatar ATen build");
#endif
}

void select_prefill_topk_block_indices_(
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &cu_seqlen_ks,
    const Tensor &cu_seqlen_ke) {
    require_iluvatar(logits, "select_prefill_topk_block_indices");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_indices, logits, cu_seqlen_ks, cu_seqlen_ke);
    require_i32(topk_indices, "topk_indices");
    require_i32(cu_seqlen_ks, "cu_seqlen_ks");
    require_i32(cu_seqlen_ke, "cu_seqlen_ke");
    if (defer_if_recording([topk_indices, logits, cu_seqlen_ks, cu_seqlen_ke] {
            select_prefill_topk_block_indices_(
                topk_indices, logits, cu_seqlen_ks, cu_seqlen_ke);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    auto out_at = adaptor::to_aten_tensor(topk_indices);
    auto logits_at = adaptor::to_aten_tensor(logits);
    auto ks_at = adaptor::to_aten_tensor(cu_seqlen_ks);
    auto ke_at = adaptor::to_aten_tensor(cu_seqlen_ke);
    adaptor::vllm_iluvatar::select_prefill_topk_block_indices(logits_at, ks_at, ke_at, out_at);
    return;
#else
    throw std::runtime_error("select_prefill_topk_block_indices requires an Iluvatar ATen build");
#endif
}

void select_decode_topk_block_indices_(
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &seq_lens) {
    require_iluvatar(logits, "select_decode_topk_block_indices");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_indices, logits, seq_lens);
    require_i32(topk_indices, "topk_indices");
    require_i32(seq_lens, "seq_lens");
    if (defer_if_recording([topk_indices, logits, seq_lens] {
            select_decode_topk_block_indices_(topk_indices, logits, seq_lens);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    auto out_at = adaptor::to_aten_tensor(topk_indices);
    auto logits_at = adaptor::to_aten_tensor(logits);
    auto lens_at = adaptor::to_aten_tensor(seq_lens);
    adaptor::vllm_iluvatar::select_decode_topk_block_indices(logits_at, lens_at, out_at);
    return;
#else
    throw std::runtime_error("select_decode_topk_block_indices requires an Iluvatar ATen build");
#endif
}

void map_prefill_request_block_indices_(
    Tensor output,
    const Tensor &req_id,
    const Tensor &block_table,
    const Tensor &token_indices,
    int64_t block_size,
    bool has_prefill_workspace,
    std::optional<Tensor> prefill_workspace_request_ids,
    std::optional<Tensor> prefill_workspace_starts) {
    require_iluvatar(output, "map_prefill_request_block_indices");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, req_id, block_table, token_indices);
    if (block_size <= 0 || output->shape() != token_indices->shape()) {
        throw std::runtime_error("map_prefill_request_block_indices invalid output shape or block size");
    }
    if (defer_if_recording([output, req_id, block_table, token_indices, block_size,
                            has_prefill_workspace, prefill_workspace_request_ids,
                            prefill_workspace_starts] {
            map_prefill_request_block_indices_(
                output, req_id, block_table, token_indices, block_size,
                has_prefill_workspace, prefill_workspace_request_ids,
                prefill_workspace_starts);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    auto output_at = adaptor::to_aten_tensor(output);
    auto req_at = adaptor::to_aten_tensor(req_id);
    auto blocks_at = adaptor::to_aten_tensor(block_table);
    auto indices_at = adaptor::to_aten_tensor(token_indices);
    std::optional<at::Tensor> request_ids_at;
    std::optional<at::Tensor> starts_at;
    if (prefill_workspace_request_ids) {
        request_ids_at = adaptor::to_aten_tensor(*prefill_workspace_request_ids);
    }
    if (prefill_workspace_starts) {
        starts_at = adaptor::to_aten_tensor(*prefill_workspace_starts);
    }
    adaptor::vllm_iluvatar::map_prefill_request_block_indices(
        output_at, req_at, blocks_at, indices_at, block_size,
        has_prefill_workspace, request_ids_at, starts_at);
    return;
#else
    throw std::runtime_error("map_prefill_request_block_indices requires an Iluvatar ATen build");
#endif
}

void map_decode_request_block_indices_(
    Tensor output,
    const Tensor &req_id,
    const Tensor &block_table,
    const Tensor &token_indices,
    int64_t block_size) {
    require_iluvatar(output, "map_decode_request_block_indices");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, req_id, block_table, token_indices);
    if (block_size <= 0 || output->shape() != token_indices->shape()) {
        throw std::runtime_error("map_decode_request_block_indices invalid output shape or block size");
    }
    if (defer_if_recording([output, req_id, block_table, token_indices, block_size] {
            map_decode_request_block_indices_(
                output, req_id, block_table, token_indices, block_size);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    auto output_at = adaptor::to_aten_tensor(output);
    auto req_at = adaptor::to_aten_tensor(req_id);
    auto blocks_at = adaptor::to_aten_tensor(block_table);
    auto indices_at = adaptor::to_aten_tensor(token_indices);
    adaptor::vllm_iluvatar::map_decode_request_block_indices(
        output_at, req_at, blocks_at, indices_at, block_size);
    return;
#else
    throw std::runtime_error("map_decode_request_block_indices requires an Iluvatar ATen build");
#endif
}

void topk_indices_context_lens_(Tensor topk_lens, const Tensor &indices) {
    require_iluvatar(indices, "topk_indices_context_lens");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_lens, indices);
    require_i32(topk_lens, "topk_lens");
    require_i32(indices, "indices");
    if (defer_if_recording([topk_lens, indices] {
            topk_indices_context_lens_(topk_lens, indices);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    auto lens_at = adaptor::to_aten_tensor(topk_lens);
    auto indices_at = adaptor::to_aten_tensor(indices);
    adaptor::vllm_iluvatar::topk_indices_context_lens(lens_at, indices_at);
    return;
#else
    throw std::runtime_error("topk_indices_context_lens requires an Iluvatar ATen build");
#endif
}

void sparse_flash_mla_(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale,
    std::optional<Tensor> attn_sink) {
    require_iluvatar(output, "sparse_flash_mla");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, query, kv_cache, indices, topk_lens);
    if (query->ndim() != 3 || output->ndim() != 3 || kv_cache->ndim() != 3
        || indices->ndim() != 3 || topk_lens->ndim() != 1) {
        throw std::runtime_error("sparse_flash_mla tensor rank mismatch");
    }
    require_i32(indices, "indices");
    require_i32(topk_lens, "topk_lens");
    if (kv_cache->dtype() == DataType::U8) {
        if (attn_sink.has_value()) {
            throw std::runtime_error(
                "fp8 sparse MLA does not support attention sinks");
        }
        fp8_sparse_mla_(
            output, query, kv_cache, indices, topk_lens, scale);
        return;
    }
    if (defer_if_recording([output, query, kv_cache, indices, topk_lens, scale, attn_sink] {
            sparse_flash_mla_(
                output, query, kv_cache, indices, topk_lens, scale,
                attn_sink);
        })) {
        return;
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (!adaptor::vllm_iluvatar::sparse_flash_mla_available()) {
        throw std::runtime_error("Iluvatar sparse FlashMLA extension is unavailable");
    }
    auto output_at = adaptor::to_aten_tensor(output);
    auto query_at = adaptor::to_aten_tensor(query);
    auto cache_at = adaptor::to_aten_tensor(kv_cache);
    auto indices_at = adaptor::to_aten_tensor(indices);
    auto lens_at = adaptor::to_aten_tensor(topk_lens);
    std::optional<at::Tensor> sink_at;
    if (attn_sink) {
        sink_at = adaptor::to_aten_tensor(*attn_sink);
    }
    adaptor::vllm_iluvatar::sparse_flash_mla(
        output_at, query_at, cache_at, indices_at, lens_at, scale, sink_at);
    return;
#else
    throw std::runtime_error("sparse_flash_mla requires an Iluvatar ATen build");
#endif
}

} // namespace infinicore::op
