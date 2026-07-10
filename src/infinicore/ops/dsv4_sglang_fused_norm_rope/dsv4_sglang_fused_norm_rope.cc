#include "infinicore/ops/dsv4_sglang_fused_norm_rope.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangFusedNormRope);

Dsv4SglangFusedNormRope::Dsv4SglangFusedNormRope(Tensor kv, const Tensor &weight, const Tensor &positions, const Tensor &freqs, double eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(kv, weight, positions, freqs);
    INFINICORE_GRAPH_OP_DISPATCH(kv->device().getType(), kv, weight, positions, freqs, eps);
}

void Dsv4SglangFusedNormRope::execute(Tensor kv, const Tensor &weight, const Tensor &positions, const Tensor &freqs, double eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangFusedNormRope, kv, weight, positions, freqs, eps);
}

void dsv4_sglang_fused_norm_rope_(Tensor kv, const Tensor &weight, const Tensor &positions, const Tensor &freqs, double eps) {
    Dsv4SglangFusedNormRope::execute(kv, weight, positions, freqs, eps);
}

} // namespace infinicore::op
