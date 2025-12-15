#include "infinicore/ops/log2.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Log2::schema> &Log2::dispatcher() {
    static common::OpDispatcher<Log2::schema> dispatcher_;
    return dispatcher_;
};

void Log2::execute(Tensor output, Tensor input) {
    infinicore::context::setDevice(output->device(), true);
    dispatcher().lookup(output->device().getType())(output, input);
}

Tensor log2(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    log2_(output, input);
    return output;
}

void log2_(Tensor output, Tensor input) {
    Log2::execute(output, input);
}

} // namespace infinicore::op
