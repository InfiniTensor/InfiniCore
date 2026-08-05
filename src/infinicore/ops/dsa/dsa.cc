#include "infinicore/ops/dsa.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/fp8_sparse_mla.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

#include "../vendor_ops/vendor_ops_dispatch.hpp"

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

void require_tensor(const Tensor &tensor, const char *op_name) {
    if (!tensor) {
        throw std::runtime_error(std::string(op_name) + " expects non-empty tensors");
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
    require_tensor(q_out, "fused_deepseek_v2_indexer_postprocess");
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
    auto kernel = vendor_ops::lookup(
        vendor_ops::fused_indexer_postprocess_dispatcher(), q_out->device().getType(),
        "fused_deepseek_v2_indexer_postprocess");
    kernel(q_out, k_out, weights_out, kv_cache, slot_mapping, q, kw, norm_weight,
           norm_bias, positions, cos_sin_cache, num_cache_tokens, is_neox, eps,
           weights_scale);
}

void indexer_k_cache_(const Tensor &k, Tensor kv_cache, const Tensor &slot_mapping) {
    require_tensor(k, "indexer_k_cache");
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
    auto kernel = vendor_ops::lookup(
        vendor_ops::indexer_k_cache_dispatcher(), k->device().getType(),
        "indexer_k_cache");
    kernel(k, kv_cache, slot_mapping);
}

void indexer_k_quant_and_cache_(
    const Tensor &k,
    Tensor kv_cache,
    const Tensor &slot_mapping,
    int64_t quant_block_size,
    const std::string &scale_fmt) {
    require_tensor(k, "indexer_k_quant_and_cache");
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
    auto kernel = vendor_ops::lookup(
        vendor_ops::indexer_k_quant_and_cache_dispatcher(), k->device().getType(),
        "indexer_k_quant_and_cache");
    kernel(k, kv_cache, slot_mapping, quant_block_size, scale_fmt);
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
    require_tensor(q, "compute_block_sparse_mqa_logits");
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
    auto kernel = vendor_ops::lookup(
        vendor_ops::block_sparse_mqa_logits_dispatcher(), q->device().getType(),
        "compute_block_sparse_mqa_logits");
    kernel(logits, q, kv_cache, cu_seqlens_q, cu_seqlens_kv, block_table,
           weights, max_q_len, max_kv_len, max_context_len);
}

void select_prefill_topk_block_indices_(
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &cu_seqlen_ks,
    const Tensor &cu_seqlen_ke) {
    require_tensor(logits, "select_prefill_topk_block_indices");
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
    auto kernel = vendor_ops::lookup(
        vendor_ops::select_prefill_topk_dispatcher(), logits->device().getType(),
        "select_prefill_topk_block_indices");
    kernel(topk_indices, logits, cu_seqlen_ks, cu_seqlen_ke);
}

void select_decode_topk_block_indices_(
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &seq_lens) {
    require_tensor(logits, "select_decode_topk_block_indices");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_indices, logits, seq_lens);
    require_i32(topk_indices, "topk_indices");
    require_i32(seq_lens, "seq_lens");
    if (defer_if_recording([topk_indices, logits, seq_lens] {
            select_decode_topk_block_indices_(topk_indices, logits, seq_lens);
        })) {
        return;
    }
    auto kernel = vendor_ops::lookup(
        vendor_ops::select_decode_topk_dispatcher(), logits->device().getType(),
        "select_decode_topk_block_indices");
    kernel(topk_indices, logits, seq_lens);
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
    require_tensor(output, "map_prefill_request_block_indices");
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
    auto kernel = vendor_ops::lookup(
        vendor_ops::map_prefill_indices_dispatcher(), output->device().getType(),
        "map_prefill_request_block_indices");
    kernel(output, req_id, block_table, token_indices, block_size,
           has_prefill_workspace, prefill_workspace_request_ids,
           prefill_workspace_starts);
}

void map_decode_request_block_indices_(
    Tensor output,
    const Tensor &req_id,
    const Tensor &block_table,
    const Tensor &token_indices,
    int64_t block_size) {
    require_tensor(output, "map_decode_request_block_indices");
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
    auto kernel = vendor_ops::lookup(
        vendor_ops::map_decode_indices_dispatcher(), output->device().getType(),
        "map_decode_request_block_indices");
    kernel(output, req_id, block_table, token_indices, block_size);
}

void topk_indices_context_lens_(Tensor topk_lens, const Tensor &indices) {
    require_tensor(indices, "topk_indices_context_lens");
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_lens, indices);
    require_i32(topk_lens, "topk_lens");
    require_i32(indices, "indices");
    if (defer_if_recording([topk_lens, indices] {
            topk_indices_context_lens_(topk_lens, indices);
        })) {
        return;
    }
    auto kernel = vendor_ops::lookup(
        vendor_ops::topk_context_lens_dispatcher(), indices->device().getType(),
        "topk_indices_context_lens");
    kernel(topk_lens, indices);
}

void sparse_flash_mla_(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale,
    std::optional<Tensor> attn_sink) {
    require_tensor(output, "sparse_flash_mla");
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
    auto kernel = vendor_ops::lookup(
        vendor_ops::sparse_flash_mla_dispatcher(), output->device().getType(),
        "sparse_flash_mla");
    kernel(output, query, kv_cache, indices, topk_lens, scale, attn_sink);
}

} // namespace infinicore::op
