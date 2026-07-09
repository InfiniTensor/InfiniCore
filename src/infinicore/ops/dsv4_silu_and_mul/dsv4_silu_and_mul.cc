#include "infinicore/ops/dsv4_silu_and_mul.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SiluAndMul);

Dsv4SiluAndMul::Dsv4SiluAndMul(Tensor y, const Tensor &gate, const Tensor &up) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, gate, up);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, gate, up);
}

void Dsv4SiluAndMul::execute(Tensor y, const Tensor &gate, const Tensor &up) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SiluAndMul, y, gate, up);
}

Tensor dsv4_silu_and_mul(const Tensor &gate, const Tensor &up) {
    auto y = Tensor::empty(gate->shape(), gate->dtype(), gate->device());
    dsv4_silu_and_mul_(y, gate, up);
    return y;
}

void dsv4_silu_and_mul_(Tensor y, const Tensor &gate, const Tensor &up) {
    Dsv4SiluAndMul::execute(y, gate, up);
}

} // namespace infinicore::op
