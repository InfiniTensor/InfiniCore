#include "infinicore/ops/elu.hpp"

namespace infinicore::op {

common::OpDispatcher<Elu::schema> &Elu::dispatcher() {
    static common::OpDispatcher<Elu::schema> dispatcher_;
    return dispatcher_;
}

void Elu::execute(Tensor output, Tensor input, float alpha) {
    dispatcher().lookup(context::getDevice().getType())(output, input, alpha);
}

Tensor elu(Tensor input, float alpha) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    elu_(output, input, alpha);
    return output;
}

void elu_(Tensor output, Tensor input, float alpha) {
    Elu::execute(output, input, alpha);
}

} // namespace infinicore::op
