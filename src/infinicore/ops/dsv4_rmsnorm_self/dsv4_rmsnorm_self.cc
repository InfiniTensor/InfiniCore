#include "infinicore/ops/dsv4_rmsnorm_self.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4RMSNormSelf);

Dsv4RMSNormSelf::Dsv4RMSNormSelf(Tensor y, const Tensor &x, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, epsilon);
}

void Dsv4RMSNormSelf::execute(Tensor y, const Tensor &x, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4RMSNormSelf, y, x, epsilon);
}

Tensor dsv4_rmsnorm_self(const Tensor &x, float epsilon) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    dsv4_rmsnorm_self_(y, x, epsilon);
    return y;
}

void dsv4_rmsnorm_self_(Tensor y, const Tensor &x, float epsilon) {
    Dsv4RMSNormSelf::execute(y, x, epsilon);
}

} // namespace infinicore::op
