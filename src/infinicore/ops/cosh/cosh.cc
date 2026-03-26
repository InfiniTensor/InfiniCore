#include "infinicore/ops/cosh.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Cosh::schema> &Cosh::dispatcher() {
    static common::OpDispatcher<Cosh::schema> dispatcher_;
    return dispatcher_;
};

void Cosh::execute(Tensor y, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x);
}

Tensor cosh(Tensor x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    cosh_(y, x);
    return y;
}

void cosh_(Tensor y, Tensor x) {
    Cosh::execute(y, x);
}

} // namespace infinicore::op