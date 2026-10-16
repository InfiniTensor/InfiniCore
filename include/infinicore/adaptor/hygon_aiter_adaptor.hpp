#pragma once

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include <ATen/ATen.h>

#include <optional>

namespace infinicore::adaptor::hygon_aiter {

bool dynamic_per_token_scaled_quant_available();
bool rotary_embedding_available();
bool grouped_topk_available();
bool fused_moe_w8a8_available();

void dynamic_per_token_scaled_quant(at::Tensor &output,
                                    const at::Tensor &input,
                                    at::Tensor &scales);

void rotary_embedding(const at::Tensor &positions,
                      at::Tensor &query,
                      at::Tensor &key,
                      int64_t head_size,
                      const at::Tensor &cos_cache,
                      const at::Tensor &sin_cache,
                      bool is_neox,
                      bool is_nope_first);

void grouped_topk(at::Tensor &topk_weights,
                  at::Tensor &topk_ids,
                  const at::Tensor &scores,
                  int64_t num_expert_group,
                  int64_t topk_group,
                  bool renormalize,
                  const std::optional<at::Tensor> &bias,
                  bool is_softmax);

void fused_moe_w8a8(at::Tensor &output,
                     const at::Tensor &input,
                     const at::Tensor &w1,
                     const at::Tensor &w2,
                     const at::Tensor &topk_weights,
                     const at::Tensor &topk_ids,
                     const at::Tensor &w1_scale,
                     const at::Tensor &w2_scale);

} // namespace infinicore::adaptor::hygon_aiter

#endif
