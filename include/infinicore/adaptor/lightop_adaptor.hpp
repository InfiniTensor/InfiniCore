#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>

namespace infinicore::adaptor::lightop {

bool available();

void preload_moe_w16a16_ops();

void preload_moe_w8a8_ops();

void preload_moe_align();

void preload_moe_w8a8_marlin_asm();

void preload_silu_and_mul();

void preload_rms_rotary_embedding();

void fuse_silu_and_mul(
    at::Tensor &input,
    at::Tensor &output);

void rms_rotary_embedding_fuse(
    at::Tensor &positions,
    at::Tensor &query,
    at::Tensor &key,
    int64_t head_size,
    at::Tensor &cos_sin_cache,
    bool is_neox,
    at::Tensor q_weight,
    at::Tensor k_weight,
    const std::optional<at::Tensor> &q_bias = std::nullopt,
    const std::optional<at::Tensor> &k_bias = std::nullopt,
    double epsilon = 1e-6);

void moe_sum(
    at::Tensor &input,
    at::Tensor &output,
    const std::optional<at::Tensor> &bias = std::nullopt,
    const std::optional<at::Tensor> &expert_mask = std::nullopt,
    const std::optional<at::Tensor> &local_num_tokens = std::nullopt,
    float factor = 1.0f,
    int expect_m = -1);

void moe_align_block_size(
    at::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    at::Tensor sorted_token_ids,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    const std::optional<at::Tensor> &expert_map = std::nullopt,
    const std::optional<at::Tensor> &expert_mask = std::nullopt,
    const std::optional<at::Tensor> &num_local_tokens = std::nullopt,
    bool is_ep = false,
    bool fuse_fill = true);
void moe_gemm_marlin_w16a16(
    at::Tensor input,
    at::Tensor b_qweight,
    at::Tensor output,
    const std::optional<at::Tensor> &topk_weights,
    at::Tensor sorted_token_ids,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    int64_t top_k,
    int mode,
    int delta);

void moe_gemm_marlin_w8a8(
    at::Tensor input,
    at::Tensor b_qweight,
    at::Tensor output,
    at::Tensor a_scale,
    at::Tensor b_scale,
    const std::optional<at::Tensor> &topk_weights,
    at::Tensor sorted_token_ids,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    int64_t top_k,
    int mode,
    int delta);

void fuse_silu_mul_quant(
    at::Tensor &input,
    at::Tensor &output,
    at::Tensor &scales,
    std::optional<at::Tensor> &num_local_tokens,
    int topk,
    int expect_m,
    std::optional<at::Tensor> &expert_ids);

void preload_w8a8_linear_ops();

void per_token_dynamic_quant_int8(
    at::Tensor &output,
    const at::Tensor &input,
    at::Tensor &scales,
    const at::Tensor &smooth);

void blaslt_w8a8_gemm(
    at::Tensor &output,
    const at::Tensor &a,
    const at::Tensor &b,
    const at::Tensor &scale_a,
    const at::Tensor &scale_b,
    const std::optional<at::Tensor> &bias);

} // namespace infinicore::adaptor::lightop

#endif // ENABLE_HYGON_API && ENABLE_ATEN
