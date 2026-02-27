#include "infinicore/ops/swiglu_cuda.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SwiGLUCuda);

SwiGLUCuda::SwiGLUCuda(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b);
}

void SwiGLUCuda::execute(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SwiGLUCuda, c, a, b);
}

Tensor swiglu_cuda(const Tensor &a, const Tensor &b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    swiglu_cuda_(c, a, b);
    return c;
}

void swiglu_cuda_(Tensor c, const Tensor &a, const Tensor &b) {
    SwiGLUCuda::execute(c, a, b);
}

} // namespace infinicore::op
