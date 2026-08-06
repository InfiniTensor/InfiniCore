#pragma once

#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

// A16 x prepared AWQ W4 matrix multiplication. qweight stores two unsigned
// int4 values per byte in [K, N / 2] order. qzeros and scales use group size
// 64 and have shapes [K / 64, N / 2] and [K / 64, N], respectively.
Tensor scaled_mm_w4a16_awq(const Tensor &input, const Tensor &qweight,
                           const Tensor &qzeros, const Tensor &scales,
                           std::optional<Tensor> bias = std::nullopt);
void scaled_mm_w4a16_awq_(Tensor out, const Tensor &input, const Tensor &qweight,
                          const Tensor &qzeros, const Tensor &scales,
                          std::optional<Tensor> bias = std::nullopt);

} // namespace infinicore::op
