#include "infinicore/ops/log1p.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Log1p::schema> &Log1p::dispatcher() {
    static common::OpDispatcher<Log1p::schema> dispatcher_;
    return dispatcher_;
};

void Log1p::execute(Tensor output, Tensor input) {
    infinicore::context::setDevice(input->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Log1p implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor log1p(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    log1p_(output, input);
    return output;
}

void log1p_(Tensor output, Tensor input) {
    Log1p::execute(output, input);
}
} // namespace infinicore::op
