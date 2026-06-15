#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(AwqMarlinGemm, Tensor, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, int64_t, bool, bool, bool, bool);

void awq_marlin_gemm_(Tensor c, const Tensor &a, const Tensor &b, Tensor &b_bias, Tensor &b_scales, Tensor &a_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float);

} // namespace infinicore::op
