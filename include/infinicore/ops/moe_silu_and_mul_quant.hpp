#pragma once
#include "../tensor.hpp"
#include <cstdint>
#include <optional>

namespace infinicore::op {
// input shape [M, 2N]. format: 0=normal fp output, 1=quant int8+scale, 2=packed int8+scale.
void moe_silu_and_mul_quant_(Tensor output, std::optional<Tensor> output_scale, const Tensor &input, int64_t format = 0);
} // namespace infinicore::op
