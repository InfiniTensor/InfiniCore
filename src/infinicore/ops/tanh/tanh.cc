#include "infinicore/ops/tanh.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Tanh);

Tanh::Tanh(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input);
}

void Tanh::execute(Tensor output, const Tensor &input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Tanh, output, input);
}

Tensor tanh(const Tensor &input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    tanh_(output, input);
    return output;
}

void tanh_(Tensor output, const Tensor &input) {
    Tanh::execute(output, input);
}

} // namespace infinicore::op
