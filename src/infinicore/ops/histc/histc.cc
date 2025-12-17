#include "infinicore/ops/histc.hpp"
#include <iostream>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Histc::schema> &Histc::dispatcher() {
    static common::OpDispatcher<Histc::schema> dispatcher_;
    return dispatcher_;
};

void Histc::execute(Tensor input, Tensor output, size_t bins, double min, double max) {
    infinicore::context::setDevice(input->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Histc implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(input, output, bins, min, max);
}

Tensor histc(Tensor input, size_t bins, double min, double max) {
    auto output = Tensor::empty(Shape{
                                    bins,
                                },
                                input->dtype(), input->device());
    histc_(input, output, bins, min, max);
    return output;
}

void histc_(Tensor input, Tensor output, size_t bins, double min, double max) {
    Histc::execute(input, output, bins, min, max);
}
} // namespace infinicore::op
