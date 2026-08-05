#pragma once

#include "../tensor.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op {

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
    double weights_scale);

void indexer_k_cache_(const Tensor &k, Tensor kv_cache, const Tensor &slot_mapping);

void indexer_k_quant_and_cache_(
    const Tensor &k,
    Tensor kv_cache,
    const Tensor &slot_mapping,
    int64_t quant_block_size = 128,
    const std::string &scale_fmt = "ue8m0");

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
    int64_t max_context_len);

void select_prefill_topk_block_indices_(
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &cu_seqlen_ks,
    const Tensor &cu_seqlen_ke);

void select_decode_topk_block_indices_(
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &seq_lens);

void map_prefill_request_block_indices_(
    Tensor output,
    const Tensor &req_id,
    const Tensor &block_table,
    const Tensor &token_indices,
    int64_t block_size,
    bool has_prefill_workspace = false,
    std::optional<Tensor> prefill_workspace_request_ids = std::nullopt,
    std::optional<Tensor> prefill_workspace_starts = std::nullopt);

void map_decode_request_block_indices_(
    Tensor output,
    const Tensor &req_id,
    const Tensor &block_table,
    const Tensor &token_indices,
    int64_t block_size);

void topk_indices_context_lens_(Tensor topk_lens, const Tensor &indices);

void sparse_flash_mla_(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale,
    std::optional<Tensor> attn_sink = std::nullopt);

} // namespace infinicore::op
