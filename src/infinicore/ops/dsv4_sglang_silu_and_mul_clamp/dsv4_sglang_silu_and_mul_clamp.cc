#include "infinicore/ops/dsv4_sglang_silu_and_mul_clamp.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangSiluAndMulClamp);

Dsv4SglangSiluAndMulClamp::Dsv4SglangSiluAndMulClamp(Tensor output, const Tensor &input, double limit) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input, limit);
}

void Dsv4SglangSiluAndMulClamp::execute(Tensor output, const Tensor &input, double limit) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangSiluAndMulClamp, output, input, limit);
}

void dsv4_sglang_silu_and_mul_clamp_(Tensor output, const Tensor &input, double limit) {
    Dsv4SglangSiluAndMulClamp::execute(output, input, limit);
}

} // namespace infinicore::op
