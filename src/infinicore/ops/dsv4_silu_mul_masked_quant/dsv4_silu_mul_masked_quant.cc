#include "infinicore/ops/dsv4_silu_mul_masked_quant.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SiluMulMaskedQuant);

Dsv4SiluMulMaskedQuant::Dsv4SiluMulMaskedQuant(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up, const Tensor &mask) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, scale, gate, up, mask);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, scale, gate, up, mask);
}

void Dsv4SiluMulMaskedQuant::execute(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up, const Tensor &mask) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SiluMulMaskedQuant, q, scale, gate, up, mask);
}

void dsv4_silu_mul_masked_quant_(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up, const Tensor &mask) {
    Dsv4SiluMulMaskedQuant::execute(q, scale, gate, up, mask);
}

} // namespace infinicore::op
