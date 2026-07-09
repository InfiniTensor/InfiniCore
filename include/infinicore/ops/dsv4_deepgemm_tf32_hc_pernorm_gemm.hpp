#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4DeepgemmTf32HcPernormGemm,
                          const Tensor &,
                          const Tensor &,
                          Tensor,
                          Tensor,
                          int64_t);

void dsv4_deepgemm_tf32_hc_pernorm_gemm_(const Tensor &a, const Tensor &b, Tensor d, Tensor sqr_sum, int64_t num_splits);

} // namespace infinicore::op
