#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <infiniccl.h>
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    LinearAllReduce,
    Tensor,
    const Tensor &,
    const Tensor &,
    const std::optional<Tensor> &,
    infinicclComm_t);

Tensor linear_allreduce(
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator);

Tensor linear_allreduce_packed(
    Tensor input,
    Tensor packed_weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator);

} // namespace infinicore::op
