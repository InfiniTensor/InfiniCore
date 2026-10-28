#include "infinicore/ops/qk_rms_norm_rope.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(QKRMSNormRoPE);
QKRMSNormRoPE::QKRMSNormRoPE(Tensor q_out,
                             Tensor k_out,
                             const Tensor &q,
                             const Tensor &k,
                             const Tensor &q_weight,
                             const Tensor &k_weight,
                             const Tensor &cos,
                             const Tensor &sin,
                             const Tensor &positions, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon);
}
void QKRMSNormRoPE::execute(Tensor q_out,
                            Tensor k_out,
                            const Tensor &q,
                            const Tensor &k,
                            const Tensor &q_weight,
                            const Tensor &k_weight,
                            const Tensor &cos,
                            const Tensor &sin,
                            const Tensor &positions, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(QKRMSNormRoPE, q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon);
}
void qk_rms_norm_rope_(Tensor q_out,
                       Tensor k_out,
                       const Tensor &q,
                       const Tensor &k,
                       const Tensor &q_weight,
                       const Tensor &k_weight,
                       const Tensor &cos,
                       const Tensor &sin,
                       const Tensor &positions, float epsilon) { QKRMSNormRoPE::execute(q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon); }
} // namespace infinicore::op
