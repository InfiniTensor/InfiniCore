#include "infinicore/ops/sqrt.hpp"

namespace infinicore::op {

common::OpDispatcher<Sqrt::schema> &Sqrt::dispatcher() {
    static common::OpDispatcher<Sqrt::schema> dispatcher_;
    return dispatcher_;
}

void Sqrt::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

Tensor sqrt(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    sqrt_(output, input);
    return output;
}

void sqrt_(Tensor output, Tensor input) {
    Sqrt::execute(output, input);
}

} // namespace infinicore::op