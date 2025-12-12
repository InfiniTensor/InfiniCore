#include "infinicore/ops/erfinv.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Erfinv::schema> &Erfinv::dispatcher() {
    static common::OpDispatcher<Erfinv::schema> dispatcher_;
    return dispatcher_;
};

void Erfinv::execute(Tensor output, Tensor input) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Erfinv implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor erfinv(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    erfinv_(output, input);
    return output;
}

void erfinv_(Tensor output, Tensor input) {
    Erfinv::execute(output, input);
}
} // namespace infinicore::op

