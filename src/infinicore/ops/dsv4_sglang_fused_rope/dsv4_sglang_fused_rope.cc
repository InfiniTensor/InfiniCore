#include "infinicore/ops/dsv4_sglang_fused_rope.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangFusedRope);

Dsv4SglangFusedRope::Dsv4SglangFusedRope(Tensor q, const Tensor &freqs_cis, const Tensor &positions, bool inverse) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, freqs_cis, positions);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, freqs_cis, positions, inverse);
}

void Dsv4SglangFusedRope::execute(Tensor q, const Tensor &freqs_cis, const Tensor &positions, bool inverse) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangFusedRope, q, freqs_cis, positions, inverse);
}

void dsv4_sglang_fused_rope_(Tensor q, const Tensor &freqs_cis, const Tensor &positions, bool inverse) {
    Dsv4SglangFusedRope::execute(q, freqs_cis, positions, inverse);
}

} // namespace infinicore::op
