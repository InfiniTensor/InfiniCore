#include "infinicore/ops/dsv4_per_token_quant_int8.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4PerTokenQuantInt8);

namespace {
Size leading_rows(const Tensor &x) {
    Size rows = 1;
    const auto &shape = x->shape();
    for (Size i = 0; i + 1 < shape.size(); ++i) {
        rows *= shape[i];
    }
    return rows;
}
} // namespace

Dsv4PerTokenQuantInt8::Dsv4PerTokenQuantInt8(Tensor q, Tensor scale, const Tensor &x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, scale, x);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, scale, x);
}

void Dsv4PerTokenQuantInt8::execute(Tensor q, Tensor scale, const Tensor &x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4PerTokenQuantInt8, q, scale, x);
}

std::pair<Tensor, Tensor> dsv4_per_token_quant_int8(const Tensor &x) {
    auto q = Tensor::empty(x->shape(), DataType::I8, x->device());
    auto scale = Tensor::empty({leading_rows(x), 1}, DataType::F32, x->device());
    dsv4_per_token_quant_int8_(q, scale, x);
    return {q, scale};
}

void dsv4_per_token_quant_int8_(Tensor q, Tensor scale, const Tensor &x) {
    Dsv4PerTokenQuantInt8::execute(q, scale, x);
}

} // namespace infinicore::op
