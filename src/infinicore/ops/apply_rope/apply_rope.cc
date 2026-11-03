#include "infinicore/ops/apply_rope.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(ApplyRoPE);
ApplyRoPE::ApplyRoPE(Tensor q, Tensor k, const Tensor &cos, const Tensor &sin) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, cos, sin);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, k, cos, sin);
}
void ApplyRoPE::execute(Tensor q, Tensor k, const Tensor &cos, const Tensor &sin) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(ApplyRoPE, q, k, cos, sin);
}
void apply_rope_(Tensor q, Tensor k, const Tensor &cos, const Tensor &sin) {
    ApplyRoPE::execute(q, k, cos, sin);
}
} // namespace infinicore::op
