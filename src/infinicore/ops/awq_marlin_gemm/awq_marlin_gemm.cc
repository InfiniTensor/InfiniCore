#include "infinicore/ops/awq_marlin_gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(AwqMarlinGemm);

AwqMarlinGemm::AwqMarlinGemm(Tensor c, const Tensor &a, const Tensor &b, Tensor &b_bias, Tensor &b_scales, Tensor &a_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b, b_bias, b_scales, a_scales, global_scales, b_zeros, g_idx, perm);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b, b_bias, b_scales, a_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}
void AwqMarlinGemm::execute(Tensor c, const Tensor &a, const Tensor &b, Tensor &b_bias, Tensor &b_scales, Tensor &a_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(AwqMarlinGemm, c, a, b, b_bias, b_scales, a_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

void awq_marlin_gemm_(Tensor c, const Tensor &a, const Tensor &b, Tensor &b_bias, Tensor &b_scales, Tensor &a_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    AwqMarlinGemm::execute(c, a, b, b_bias, b_scales, a_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

} // namespace infinicore::op
