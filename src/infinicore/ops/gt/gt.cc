#include "infinicore/ops/gt.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
common::OpDispatcher<Gt::schema> &Gt::dispatcher() {
    static common::OpDispatcher<Gt::schema> dispatcher_;
    return dispatcher_;
}
void Gt::execute(Tensor input, Tensor other, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, other, output);
}
Tensor gt(Tensor input, Tensor other) {
    auto output = Tensor::empty(input->shape(), DataType::BOOL, input->device());
    gt_(input, other, output);
    return output;
}
void gt_(Tensor input, Tensor other, Tensor output) {
    Gt::execute(input, other, output);
}
} // namespace infinicore::op