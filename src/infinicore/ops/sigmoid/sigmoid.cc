#include "infinicore/ops/sigmoid.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Sigmoid::schema> &Sigmoid::dispatcher() {
    static common::OpDispatcher<Sigmoid::schema> dispatcher_;
    return dispatcher_;
};

void Sigmoid::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Sigmoid implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor sigmoid(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    sigmoid_(output, input);
    return output;
}

void sigmoid_(Tensor output, Tensor input) {
    Sigmoid::execute(output, input);
}
} // namespace infinicore::op
