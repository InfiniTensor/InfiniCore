#include "infinicore/ops/dsv4_act_quant_fp8.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4ActQuantFp8);
Dsv4ActQuantFp8::Dsv4ActQuantFp8(Tensor xq, Tensor scale, const Tensor &x, float fp8_max) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(xq, scale, x);
    INFINICORE_GRAPH_OP_DISPATCH(xq->device().getType(), xq, scale, x, fp8_max);
}
void Dsv4ActQuantFp8::execute(Tensor xq, Tensor scale, const Tensor &x, float fp8_max) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4ActQuantFp8, xq, scale, x, fp8_max);
}
void dsv4_act_quant_fp8_(Tensor xq, Tensor scale, const Tensor &x, float fp8_max) { Dsv4ActQuantFp8::execute(xq, scale, x, fp8_max); }
} // namespace infinicore::op
