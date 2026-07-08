#include "infinicore/ops/dsv4_linear_bf16_fp32.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4LinearBf16Fp32);

Dsv4LinearBf16Fp32::Dsv4LinearBf16Fp32(Tensor y, const Tensor &x, const Tensor &w) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, w);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, w);
}

void Dsv4LinearBf16Fp32::execute(Tensor y, const Tensor &x, const Tensor &w) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4LinearBf16Fp32, y, x, w);
}

Tensor dsv4_linear_bf16_fp32(const Tensor &x, const Tensor &w) {
    auto x_shape = x->shape();
    auto w_shape = w->shape();
    INFINICORE_ASSERT(x_shape.size() == 2 && w_shape.size() == 2 && x_shape[1] == w_shape[1]);
    auto y = Tensor::empty({x_shape[0], w_shape[0]}, DataType::F32, x->device());
    dsv4_linear_bf16_fp32_(y, x, w);
    return y;
}

void dsv4_linear_bf16_fp32_(Tensor y, const Tensor &x, const Tensor &w) {
    Dsv4LinearBf16Fp32::execute(y, x, w);
}

} // namespace infinicore::op
