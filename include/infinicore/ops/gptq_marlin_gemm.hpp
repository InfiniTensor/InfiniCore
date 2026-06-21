#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(GptqMarlinGemm, Tensor, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, int64_t, bool, bool, bool, bool);
INFINICORE_GRAPH_OP_CLASS(GptqMarlinGemmWithWorkspace, Tensor, Tensor, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, Tensor &, int64_t, bool, bool, bool, bool);

void gptq_marlin_gemm_(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float);

size_t gptq_marlin_gemm_workspace_size(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm);
void gptq_marlin_gemm_with_workspace_(Tensor workspace, Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float);
void gptq_marlin_gemm_with_workspace_direct_(Tensor workspace, Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float);
} // namespace infinicore::op
