#pragma once

#include "../tensor.hpp"
#include <optional>

namespace infinicore::op {
void w16a16_group_gemm_(Tensor out,
                        const Tensor &input,
                        const Tensor &weight,
                        const Tensor &tokens_per_experts,
                        std::optional<Tensor> sorted_token_ids = std::nullopt,
                        std::optional<Tensor> bias = std::nullopt,
                        bool trans_weight = true,
                        bool is_decode = false);
} // namespace infinicore::op
