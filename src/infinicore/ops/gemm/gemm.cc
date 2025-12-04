#include "infinicore/ops/gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Gemm::schema> &Gemm::dispatcher() {
    static common::OpDispatcher<Gemm::schema> dispatcher_;
    return dispatcher_;
};

void Gemm::execute(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b, alpha, beta);
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
