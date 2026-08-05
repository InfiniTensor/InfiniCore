#pragma once

#include "../tensor.hpp"
#include <optional>

namespace infinicore::op {
void scaled_mm_w4a8_(Tensor out, const Tensor &a, const Tensor &b, const Tensor &a_scales, const Tensor &b_scales, std::optional<Tensor> bias = std::nullopt, bool trans_weight = false);
Tensor scaled_mm_w4a8(const Tensor &a, const Tensor &b, const Tensor &a_scales, const Tensor &b_scales, std::optional<Tensor> bias = std::nullopt, bool trans_weight = false);
} // namespace infinicore::op
