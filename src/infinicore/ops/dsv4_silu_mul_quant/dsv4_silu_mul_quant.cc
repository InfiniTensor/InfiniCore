#include "infinicore/ops/dsv4_silu_mul_quant.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SiluMulQuant);
Dsv4SiluMulQuant::Dsv4SiluMulQuant(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, scale, gate, up);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, scale, gate, up);
}
void Dsv4SiluMulQuant::execute(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SiluMulQuant, q, scale, gate, up);
}
void dsv4_silu_mul_quant_(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up) { Dsv4SiluMulQuant::execute(q, scale, gate, up); }
} // namespace infinicore::op
