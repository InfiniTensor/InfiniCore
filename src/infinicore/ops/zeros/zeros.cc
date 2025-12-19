#include "infinicore/ops/zeros.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Zeros::schema> &Zeros::dispatcher() {
    static common::OpDispatcher<Zeros::schema> dispatcher_;
    return dispatcher_;
};

void Zeros::execute(Tensor output) {
    context::setDevice(output->device());
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Zeros implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output);
}

void zeros_(Tensor output) {
    Zeros::execute(output);
}
} // namespace infinicore::op
