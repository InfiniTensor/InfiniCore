#include "infinicore/ops/dsv4_sglang_rmsnorm.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangRmsnorm);

Dsv4SglangRmsnorm::Dsv4SglangRmsnorm(Tensor output, const Tensor &input, double eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input, eps);
}

void Dsv4SglangRmsnorm::execute(Tensor output, const Tensor &input, double eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangRmsnorm, output, input, eps);
}

void dsv4_sglang_rmsnorm_(Tensor output, const Tensor &input, double eps) {
    Dsv4SglangRmsnorm::execute(output, input, eps);
}

} // namespace infinicore::op
