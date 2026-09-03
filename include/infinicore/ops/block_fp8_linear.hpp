#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(BlockFP8Linear,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

Tensor block_fp8_linear(const Tensor &input,
                        const Tensor &weight,
                        const Tensor &weight_scale);

void block_fp8_linear_(Tensor output,
                       const Tensor &input,
                       const Tensor &weight,
                       const Tensor &weight_scale);

} // namespace infinicore::op
