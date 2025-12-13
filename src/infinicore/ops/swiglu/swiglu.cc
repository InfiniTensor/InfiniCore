#include "infinicore/ops/swiglu.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<SwiGLU::schema> &SwiGLU::dispatcher() {
    static common::OpDispatcher<SwiGLU::schema> dispatcher_;
    return dispatcher_;
};

void SwiGLU::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    auto device_type = c->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No SwiGLU implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(c, a, b);
}

Tensor swiglu(Tensor a, Tensor b) {
    Shape shape = a->shape();
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    swiglu_(c, a, b);
    return c;
}

void swiglu_(Tensor c, Tensor a, Tensor b) {
    SwiGLU::execute(c, a, b);
}
} // namespace infinicore::op
