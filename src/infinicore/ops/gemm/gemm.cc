#include "infinicore/ops/gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Gemm);

Gemm::Gemm(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b, alpha, beta);
}

void Gemm::execute(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Gemm, c, a, b, alpha, beta);
}

Tensor gemm(Tensor a, Tensor b, float alpha, float beta) {
    Shape shape = a->shape();
    Size size = a->ndim();
    shape[size - 1] = b->size(size - 1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    gemm_(c, a, b, alpha, beta);
    return c;
}

void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    Gemm::execute(c, a, b, alpha, beta);
}

} // namespace infinicore::op
