#include "infinicore/ops/gptq_marlin_gemm.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GptqMarlinGemm);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GptqMarlinGemmWithWorkspace);

GptqMarlinGemm::GptqMarlinGemm(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b, b_scales, global_scales, b_zeros, g_idx, perm);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, a, b, b_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}
void GptqMarlinGemm::execute(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GptqMarlinGemm, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

void gptq_marlin_gemm_(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    GptqMarlinGemm::execute(out, a, b, b_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

GptqMarlinGemmWithWorkspace::GptqMarlinGemmWithWorkspace(Tensor workspace, Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(workspace, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), workspace, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

void GptqMarlinGemmWithWorkspace::execute(Tensor workspace, Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GptqMarlinGemmWithWorkspace, workspace, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

void gptq_marlin_gemm_with_workspace_(Tensor workspace, Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    if (context::isGraphRecording()) {
        GptqMarlinGemmWithWorkspace::execute(workspace, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
    } else {
        gptq_marlin_gemm_with_workspace_direct_(workspace, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
    }
}

} // namespace infinicore::op
