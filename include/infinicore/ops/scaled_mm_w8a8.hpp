#pragma once
#include "infinicore/tensor.hpp"
#include <optional>

namespace infinicore::op {
Tensor scaled_mm_w8a8(const Tensor &a, const Tensor &b,
                      const Tensor &a_scales, const Tensor &b_scales,
                      std::optional<Tensor> bias = std::nullopt, bool trans_weight = true);
void scaled_mm_w8a8_(Tensor out, const Tensor &a, const Tensor &b,
                     const Tensor &a_scales, const Tensor &b_scales,
                     std::optional<Tensor> bias = std::nullopt, bool trans_weight = true);
} // namespace infinicore::op
