#pragma once

#include "common/op.hpp"
#include <infiniccl.h>
#include <optional>

namespace infinicore::op {

Tensor linear_allreduce(
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclReduceOp_t op,
    infinicclComm_t communicator);

void linear_allreduce_(
    Tensor output,
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclReduceOp_t op,
    infinicclComm_t communicator);

} // namespace infinicore::op
