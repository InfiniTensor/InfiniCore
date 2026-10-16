#pragma once

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include <ATen/ATen.h>

#include <optional>
#include <string>

namespace infinicore::adaptor::hygon_lightop {

bool fused_quantize_and_store_mla_kv_cache_available();
bool top_k_per_row_available();
bool convert_req_index_to_global_index_available();

void fused_quantize_and_store_mla_kv_cache(
    at::Tensor kv_c,
    at::Tensor k_pe,
    at::Tensor kv_cache,
    at::Tensor slot_mapping,
    const std::string &kv_cache_dtype,
    float scale);

void top_k_per_row_prefill(const at::Tensor &logits,
                           const at::Tensor &row_starts,
                           const at::Tensor &row_ends,
                           at::Tensor &topk_indices);

void top_k_per_row_decode(const at::Tensor &logits,
                          const at::Tensor &seq_lens,
                          at::Tensor &topk_indices);

at::Tensor convert_req_index_to_global_index(
    at::Tensor req_id,
    at::Tensor block_table,
    at::Tensor token_indices,
    int block_size,
    bool has_prefill_workspace,
    std::optional<at::Tensor> prefill_workspace_request_ids = std::nullopt,
    std::optional<at::Tensor> prefill_workspace_starts = std::nullopt);

} // namespace infinicore::adaptor::hygon_lightop

#endif
