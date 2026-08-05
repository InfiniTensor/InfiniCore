#include "infinicore/ops/silu.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Silu);

Silu::Silu(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input);
}

void Silu::execute(Tensor output, Tensor input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Silu, output, input);
}

Tensor silu(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    silu_(output, input);
    return output;
}

void silu_(Tensor output, Tensor input) {
    Silu::execute(output, input);
}
} // namespace infinicore::op
