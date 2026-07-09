#include "infinicore/ops/dsv4_per_token_group_quant_int8.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4PerTokenGroupQuantInt8);

Dsv4PerTokenGroupQuantInt8::Dsv4PerTokenGroupQuantInt8(Tensor q, Tensor scale, const Tensor &x, int group_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, scale, x);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, scale, x, group_size);
}

void Dsv4PerTokenGroupQuantInt8::execute(Tensor q, Tensor scale, const Tensor &x, int group_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4PerTokenGroupQuantInt8, q, scale, x, group_size);
}

void dsv4_per_token_group_quant_int8_(Tensor q, Tensor scale, const Tensor &x, int group_size) {
    Dsv4PerTokenGroupQuantInt8::execute(q, scale, x, group_size);
}

} // namespace infinicore::op
