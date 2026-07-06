#include "infinicore/ops/unweighted_rms_norm.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(UnweightedRMSNorm);

UnweightedRMSNorm::UnweightedRMSNorm(Tensor y, const Tensor &x, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, epsilon);
}

void UnweightedRMSNorm::execute(Tensor y, const Tensor &x, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(UnweightedRMSNorm, y, x, epsilon);
}

Tensor unweighted_rms_norm(const Tensor &x, float epsilon) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    unweighted_rms_norm_(y, x, epsilon);
    return y;
}

void unweighted_rms_norm_(Tensor y, const Tensor &x, float epsilon) {
    UnweightedRMSNorm::execute(y, x, epsilon);
}

} // namespace infinicore::op
