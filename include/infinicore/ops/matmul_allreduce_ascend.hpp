#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include "infinicore.h"

#include <infiniccl.h>

namespace infinicore::op {

// Ascend CANN MC2 bridge matching torch_npu.npu_mm_all_reduce_base:
// output = all_reduce(input @ weight_transposed).
INFINICORE_GRAPH_OP_CLASS(
    MatmulAllReduceAscend,
    Tensor,
    const Tensor &,
    const Tensor &,
    infinicclComm_t);

__export Tensor matmul_allreduce_ascend(
    const Tensor &input,
    const Tensor &weight_transposed,
    infinicclComm_t communicator);

} // namespace infinicore::op
