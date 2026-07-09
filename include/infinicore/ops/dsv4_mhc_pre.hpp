#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4MhcPre,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float);

void dsv4_mhc_pre_(Tensor output, const Tensor &input, const Tensor &scale, const Tensor &base, float eps = 1e-6f);

} // namespace infinicore::op
