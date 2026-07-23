#pragma once
#include "../tensor.hpp"
#include <cstdint>

namespace infinicore::op {
void moe_argsort_bincount_with_inv_pos_(Tensor tokens_per_experts, Tensor sorted_indices, Tensor inv_pos, const Tensor &topk_ids, int64_t num_experts);
}
