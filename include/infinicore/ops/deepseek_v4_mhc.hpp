#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MHCParams,
                          Tensor,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          size_t,
                          float);

std::tuple<Tensor, Tensor, Tensor> deepseek_v4_mhc_params(
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    size_t sinkhorn_iters,
    float epsilon);

void deepseek_v4_mhc_params_(
    Tensor pre,
    Tensor post,
    Tensor comb,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    size_t sinkhorn_iters,
    float epsilon);

} // namespace infinicore::op
