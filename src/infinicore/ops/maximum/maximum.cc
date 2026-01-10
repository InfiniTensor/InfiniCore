#include "infinicore/ops/maximum.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Maximum::schema> &Maximum::dispatcher() {
    static common::OpDispatcher<Maximum::schema> dispatcher_;
    return dispatcher_;
};

void Maximum::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor maximum(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    maximum_(c, a, b);
    return c;
}

void maximum_(Tensor c, Tensor a, Tensor b) {
    Maximum::execute(c, a, b);
}

} // namespace infinicore::op
