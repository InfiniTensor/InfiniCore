#include "infinicore/ops/erfc.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Erfc::schema> &Erfc::dispatcher() {
    static common::OpDispatcher<Erfc::schema> dispatcher_;
    return dispatcher_;
};

void Erfc::execute(Tensor output, Tensor input) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Erfc implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor erfc(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    erfc_(output, input);
    return output;
}

void erfc_(Tensor output, Tensor input) {
    Erfc::execute(output, input);
}
} // namespace infinicore::op

