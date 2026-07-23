#pragma once
#include "../tensor.hpp"
#include <cstdint>
#include <optional>

namespace infinicore::op {
// format: 0=normal, 1=quant, 2=packed. group_size is used by quant/packed paths.
void moe_expand_input_with_inv_pos_(Tensor expand_states, std::optional<Tensor> expand_scales, const Tensor &hidden_states, const Tensor &inv_pos, int64_t top_k, int64_t group_size = 128, int64_t format = 0);
} // namespace infinicore::op
