#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MHCPost,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

Tensor deepseek_v4_mhc_post(
    const Tensor &new_x,
    const Tensor &residual,
    const Tensor &post,
    const Tensor &comb);

void deepseek_v4_mhc_post_(
    Tensor y,
    const Tensor &new_x,
    const Tensor &residual,
    const Tensor &post,
    const Tensor &comb);

} // namespace infinicore::op
