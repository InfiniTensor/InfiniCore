#include "infinicore/ops/add.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Add::schema> &Add::dispatcher() {
    static common::OpDispatcher<Add::schema> dispatcher_;
    return dispatcher_;
};

void Add::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor add(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    add_(c, a, b);
    return c;
}

void add_(Tensor c, Tensor a, Tensor b) {
    Add::execute(c, a, b);
}

} // namespace infinicore::op
