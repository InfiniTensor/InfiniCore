#include "infinicore/ops/round.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Round::schema> &Round::dispatcher() {
    static common::OpDispatcher<Round::schema> dispatcher_;
    return dispatcher_;
};

void Round::execute(Tensor y, Tensor x, int decimals) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x, decimals);
}

Tensor round(Tensor x, int decimals) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    round_(y, x, decimals);
    return y;
}

void round_(Tensor y, Tensor x, int decimals) {
    Round::execute(y, x, decimals);
}

} // namespace infinicore::op