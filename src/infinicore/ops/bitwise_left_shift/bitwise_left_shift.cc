#include "infinicore/ops/bitwise_left_shift.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<BitwiseLeftShift::schema> &BitwiseLeftShift::dispatcher() {
    static common::OpDispatcher<BitwiseLeftShift::schema> dispatcher_;
    return dispatcher_;
};

void BitwiseLeftShift::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device(), true);
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor bitwise_left_shift(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    bitwise_left_shift_(c, a, b);
    return c;
}

void bitwise_left_shift_(Tensor c, Tensor a, Tensor b) {
    BitwiseLeftShift::execute(c, a, b);
}

} // namespace infinicore::op
