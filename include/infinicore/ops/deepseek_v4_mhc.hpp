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

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MHCPreCollapse,
                          Tensor,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          size_t,
                          float);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4MHCScaleMixes,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
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

std::tuple<Tensor, Tensor, Tensor> deepseek_v4_mhc_pre_collapse(
    const Tensor &x,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    size_t sinkhorn_iters,
    float epsilon);

void deepseek_v4_mhc_pre_collapse_(
    Tensor collapsed,
    Tensor post,
    Tensor comb,
    const Tensor &x,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    size_t sinkhorn_iters,
    float epsilon);

Tensor deepseek_v4_mhc_scale_mixes(
    const Tensor &x,
    const Tensor &raw_mixes,
    float epsilon);

void deepseek_v4_mhc_scale_mixes_(
    Tensor scaled,
    const Tensor &x,
    const Tensor &raw_mixes,
    float epsilon);

} // namespace infinicore::op
