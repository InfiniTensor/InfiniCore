#include "infinicore/ops/gptq_marlin_gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GptqMarlinGemm);

GptqMarlinGemm::GptqMarlinGemm(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scale, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b, b_scales, global_scale, b_zeros, g_idx, perm);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, a, b, b_scales, global_scale, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}
void GptqMarlinGemm::execute(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scale, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GptqMarlinGemm, out, a, b, b_scales, global_scale, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

void gptq_marlin_gemm_(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scale, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    GptqMarlinGemm::execute(out, a, b, b_scales, global_scale, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

} // namespace infinicore::op
