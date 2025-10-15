#include "infinicore/ops/matmul.hpp"

namespace infinicore::op {

common::OpDispatcher<Matmul::schema> &Matmul::dispatcher() {
    static common::OpDispatcher<Matmul::schema> dispatcher_;
    return dispatcher_;
};

void Matmul::execute(Tensor c, Tensor a, Tensor b) {
    dispatcher().lookup(context::getDevice().getType())(c, a, b);
}

Tensor matmul(Tensor a, Tensor b) {
    Shape shape = a->shape();
    Size size = a->ndim();
    shape[size - 1] = b->size(size - 1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    matmul_(c, a, b);
    return c;
}

void matmul_(Tensor c, Tensor a, Tensor b) {
    Matmul::execute(c, a, b);
}
} // namespace infinicore::op
