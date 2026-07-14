#pragma once
#include "../device.hpp"
#include "../tensor.hpp"
namespace infinicore::op {
void moe_topk_softmax_vllm_(Tensor topk_weights, Tensor topk_ids, Tensor token_expert_indices, const Tensor &gating_output, bool renormalize = false, const Tensor &correction_bias = Tensor());
void moe_topk_sigmoid_vllm_(Tensor topk_weights, Tensor topk_ids, Tensor token_expert_indices, const Tensor &gating_output, bool renormalize = false, const Tensor &correction_bias = Tensor());
} // namespace infinicore::op
