#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <vector>

namespace infinicore::adaptor::lightop {

bool available();

bool enabled_by_env();

void preload_basic_ops();

void preload_silu_and_mul();

void fused_rms_norm_contiguous(
    at::Tensor &out,
    at::Tensor &input,
    at::Tensor &weight,
    double epsilon);

void fuse_silu_and_mul(
    at::Tensor &input,
    at::Tensor &output);

void moe_sum(
    at::Tensor &input,
    at::Tensor &output,
    const std::optional<at::Tensor> &bias = std::nullopt,
    const std::optional<at::Tensor> &expert_mask = std::nullopt,
    const std::optional<at::Tensor> &local_num_tokens = std::nullopt,
    float factor = 1.0f,
    int expect_m = -1);

std::vector<at::Tensor> moe_fused_gate(
    at::Tensor &input,
    at::Tensor &bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor);

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

} // namespace infinicore::adaptor::lightop

#endif // ENABLE_HYGON_API && ENABLE_ATEN
