#include "infinicore/ops/erf.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Erf::schema> &Erf::dispatcher() {
    static common::OpDispatcher<Erf::schema> dispatcher_;
    return dispatcher_;
};

void Erf::execute(Tensor output, Tensor input) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Erf implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor erf(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    erf_(output, input);
    return output;
}

void erf_(Tensor output, Tensor input) {
    Erf::execute(output, input);
}
} // namespace infinicore::op

