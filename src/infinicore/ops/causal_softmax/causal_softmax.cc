#include "infinicore/ops/causal_softmax.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<CausalSoftmax::schema> &CausalSoftmax::dispatcher() {
    static common::OpDispatcher<CausalSoftmax::schema> dispatcher_;
    return dispatcher_;
};

void CausalSoftmax::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No CausalSoftmax implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor causal_softmax(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    causal_softmax_(output, input);
    return output;
}

void causal_softmax_(Tensor output, Tensor input) {
    CausalSoftmax::execute(output, input);
}
} // namespace infinicore::op
