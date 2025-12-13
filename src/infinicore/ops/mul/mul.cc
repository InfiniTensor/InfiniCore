#include "infinicore/ops/mul.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Mul::schema> &Mul::dispatcher() {
    static common::OpDispatcher<Mul::schema> dispatcher_;
    return dispatcher_;
};

void Mul::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor mul(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    mul_(c, a, b);
    return c;
}

void mul_(Tensor c, Tensor a, Tensor b) {
    Mul::execute(c, a, b);
}

} // namespace infinicore::op
