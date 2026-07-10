#include "infinicore/ops/dsv4_sglang_main_q_norm_rope.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangMainQNormRope);

Dsv4SglangMainQNormRope::Dsv4SglangMainQNormRope(Tensor output, const Tensor &input, const Tensor &freqs, const Tensor &positions, double eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, freqs, positions);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input, freqs, positions, eps);
}

void Dsv4SglangMainQNormRope::execute(Tensor output, const Tensor &input, const Tensor &freqs, const Tensor &positions, double eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangMainQNormRope, output, input, freqs, positions, eps);
}

void dsv4_sglang_main_q_norm_rope_(Tensor output, const Tensor &input, const Tensor &freqs, const Tensor &positions, double eps) {
    Dsv4SglangMainQNormRope::execute(output, input, freqs, positions, eps);
}

} // namespace infinicore::op
