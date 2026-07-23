#pragma once
#include "../tensor.hpp"
#include <optional>

namespace infinicore::op {
void moe_sum_vllm_(Tensor output, const Tensor &input, std::optional<Tensor> topk_weights = std::nullopt, std::optional<Tensor> extra_residual = std::nullopt, double routed_scale = 1.0, double residual_scale = 1.0);
}
