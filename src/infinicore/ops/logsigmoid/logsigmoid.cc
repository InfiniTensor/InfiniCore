#include "infinicore/ops/logsigmoid.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<LogSigmoid::schema> &LogSigmoid::dispatcher() {
    static common::OpDispatcher<LogSigmoid::schema> dispatcher_;
    return dispatcher_;
}

void LogSigmoid::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No LogSigmoid implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor logsigmoid(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    logsigmoid_(output, input);
    return output;
}

void logsigmoid_(Tensor output, Tensor input) {
    LogSigmoid::execute(output, input);
}
} // namespace infinicore::op
