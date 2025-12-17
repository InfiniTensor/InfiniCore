#include "infinicore/ops/log10.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Log10::schema> &Log10::dispatcher() {
    static common::OpDispatcher<Log10::schema> dispatcher_;
    return dispatcher_;
};

void Log10::execute(Tensor output, Tensor input) {
    infinicore::context::setDevice(input->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Log10 implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor log10(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    log10_(output, input);
    return output;
}

void log10_(Tensor output, Tensor input) {
    Log10::execute(output, input);
}
} // namespace infinicore::op
