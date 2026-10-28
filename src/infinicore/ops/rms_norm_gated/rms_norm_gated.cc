#include "infinicore/ops/rms_norm_gated.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(RMSNormGated);
RMSNormGated::RMSNormGated(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &weight, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x, gate, weight);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), out, x, gate, weight, epsilon);
}
void RMSNormGated::execute(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &weight, float epsilon) { INFINICORE_GRAPH_OP_RECORD_OR_RUN(RMSNormGated, out, x, gate, weight, epsilon); }
void rms_norm_gated_(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &weight, float epsilon) { RMSNormGated::execute(out, x, gate, weight, epsilon); }
Tensor rms_norm_gated(const Tensor &x, const Tensor &gate, const Tensor &weight, float epsilon) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    rms_norm_gated_(out, x, gate, weight, epsilon);
    return out;
}
} // namespace infinicore::op
