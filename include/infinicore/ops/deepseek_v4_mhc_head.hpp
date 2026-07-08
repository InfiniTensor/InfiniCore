#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MHCHeadCollapse,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float);

Tensor deepseek_v4_mhc_head_collapse(
    const Tensor &x,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    float epsilon);

void deepseek_v4_mhc_head_collapse_(
    Tensor y,
    const Tensor &x,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    float epsilon);

} // namespace infinicore::op
