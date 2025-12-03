#include "infinicore/ops/rearrange.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rearrange::schema> &Rearrange::dispatcher() {
    static common::OpDispatcher<Rearrange::schema> dispatcher_;
    return dispatcher_;
};

void Rearrange::execute(Tensor y, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x);
}

Tensor rearrange(Tensor x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    rearrange_(y, x);
    return y;
}

void rearrange_(Tensor y, Tensor x) {
    Rearrange::execute(y, x);
}
} // namespace infinicore::op
