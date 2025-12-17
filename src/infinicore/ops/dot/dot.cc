#include "infinicore/ops/dot.hpp"
#include "../../utils.hpp"
#include <iostream>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Dot::schema> &Dot::dispatcher() {
    static common::OpDispatcher<Dot::schema> dispatcher_;
    return dispatcher_;
};

void Dot::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(a->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Dot implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    if (a->ndim() != 1 || b->ndim() != 1 || c->ndim() != 0) {
        throw std::runtime_error("Dot operation only supports 1-D tensors for a and b, and 0-D tensor for c.");
    }

    func(c, a, b);
}

Tensor dot(Tensor a, Tensor b) {
    auto c = Tensor::empty(Shape{}, a->dtype(), a->device());
    dot_(c, a, b);
    return c;
}

void dot_(Tensor c, Tensor a, Tensor b) {
    Dot::execute(c, a, b);
}
} // namespace infinicore::op
