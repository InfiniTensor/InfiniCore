#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include "infinicore.h"

#include <infiniccl.h>
#include <tuple>

namespace infinicore::op {

// vLLM-Ascend vendor bridge:
//   add_out = all_reduce(input @ weight^T) + residual
//   normalized = rms_norm(add_out, gamma, epsilon)
INFINICORE_GRAPH_OP_CLASS(
    MatmulAllReduceAddRmsNormAscend,
    Tensor,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    infinicclComm_t,
    float);

__export std::tuple<Tensor, Tensor>
matmul_allreduce_add_rmsnorm_ascend(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &residual,
    const Tensor &gamma,
    infinicclComm_t communicator,
    float epsilon);

// vLLM-Ascend vendor bridge:
//   add_out = x1 + x2
//   normalized = rms_norm(add_out, gamma, epsilon)
// This directly backs RMSNorm::forward_inplace on Ascend.
__export std::tuple<Tensor, Tensor>
add_rmsnorm_ascend_vendor(
    const Tensor &x1, const Tensor &x2,
    const Tensor &gamma, float epsilon);

} // namespace infinicore::op
