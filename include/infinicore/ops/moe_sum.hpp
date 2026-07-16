#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/// Reduce MoE expert outputs: input [M, topk, H] -> [M, H] (sum over topk).
/// Thin wrapper over ``sum(..., dim={1})`` so MetaX/CUDA reuse existing kernels.
Tensor moe_sum(const Tensor &input);
void moe_sum_(Tensor out, const Tensor &input);

} // namespace infinicore::op
