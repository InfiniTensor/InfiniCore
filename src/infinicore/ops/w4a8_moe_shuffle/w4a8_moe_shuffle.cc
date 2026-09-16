#include "infinicore/ops/w4a8_moe_shuffle.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(W4A8MoeShuffle);

W4A8MoeShuffle::W4A8MoeShuffle(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input);
}

void W4A8MoeShuffle::execute(Tensor output, const Tensor &input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(W4A8MoeShuffle, output, input);
}

Tensor w4a8_moe_shuffle(const Tensor &input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    w4a8_moe_shuffle_(output, input);
    return output;
}

void w4a8_moe_shuffle_(Tensor output, const Tensor &input) {
    W4A8MoeShuffle::execute(output, input);
}

} // namespace infinicore::op
