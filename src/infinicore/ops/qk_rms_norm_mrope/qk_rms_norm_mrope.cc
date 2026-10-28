#include "infinicore/ops/qk_rms_norm_mrope.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(QKRMSNormMRoPE);
QKRMSNormMRoPE::QKRMSNormMRoPE(Tensor q_out,
                               Tensor k_out,
                               const Tensor &q,
                               const Tensor &k,
                               const Tensor &q_weight,
                               const Tensor &k_weight,
                               const Tensor &cos,
                               const Tensor &sin,
                               const Tensor &positions, float epsilon, int section_t, int section_h, int section_w, bool interleaved) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon, section_t, section_h, section_w, interleaved);
}
void QKRMSNormMRoPE::execute(Tensor q_out,
                             Tensor k_out,
                             const Tensor &q,
                             const Tensor &k,
                             const Tensor &q_weight,
                             const Tensor &k_weight,
                             const Tensor &cos,
                             const Tensor &sin,
                             const Tensor &positions, float epsilon, int section_t, int section_h, int section_w, bool interleaved) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(QKRMSNormMRoPE, q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon, section_t, section_h, section_w, interleaved);
}
void qk_rms_norm_mrope_(Tensor q_out,
                        Tensor k_out,
                        const Tensor &q,
                        const Tensor &k,
                        const Tensor &q_weight,
                        const Tensor &k_weight,
                        const Tensor &cos,
                        const Tensor &sin,
                        const Tensor &positions, float epsilon, int section_t, int section_h, int section_w, bool interleaved) { QKRMSNormMRoPE::execute(q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon, section_t, section_h, section_w, interleaved); }
} // namespace infinicore::op
