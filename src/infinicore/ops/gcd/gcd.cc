#include "infinicore/ops/gcd.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Gcd::schema> &Gcd::dispatcher() {
    static common::OpDispatcher<Gcd::schema> dispatcher_;
    return dispatcher_;
};

void Gcd::execute(Tensor input, Tensor other, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, other, output);
}

Tensor gcd(Tensor input, Tensor other) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    gcd_(input, other, output);
    return output;
}

void gcd_(Tensor input, Tensor other, Tensor output) {
    Gcd::execute(input, other, output);
}

} // namespace infinicore::op