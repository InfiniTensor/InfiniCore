#include "infinicore/ops/dsv4_fused_rope.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4FusedRope);

Dsv4FusedRope::Dsv4FusedRope(Tensor q, Tensor k, const Tensor &freq_real, const Tensor &freq_imag, bool has_k) {
    if (has_k) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, freq_real, freq_imag);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, freq_real, freq_imag);
    }
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, k, freq_real, freq_imag, has_k);
}

void Dsv4FusedRope::execute(Tensor q, Tensor k, const Tensor &freq_real, const Tensor &freq_imag, bool has_k) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4FusedRope, q, k, freq_real, freq_imag, has_k);
}

void dsv4_fused_rope_(Tensor q, Tensor k, const Tensor &freq_real, const Tensor &freq_imag, bool has_k) {
    Dsv4FusedRope::execute(q, k, freq_real, freq_imag, has_k);
}

Tensor dsv4_fused_rope(const Tensor &q, const Tensor &freq_real, const Tensor &freq_imag) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    out->copy_from(q);
    dsv4_fused_rope_(out, Tensor(), freq_real, freq_imag, false);
    return out;
}

} // namespace infinicore::op
