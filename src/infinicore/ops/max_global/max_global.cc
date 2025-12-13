#include "infinicore/ops/max_global.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<MaxGlobal::schema> &MaxGlobal::dispatcher() {
    static common::OpDispatcher<MaxGlobal::schema> dispatcher_;
    return dispatcher_;
};

void MaxGlobal::execute(Tensor input, Tensor output) {
    infinicore::context::setDevice(input->device(), true);
    dispatcher().lookup(input->device().getType())(input, output);
}

Tensor max_global(Tensor input) {
    Shape shape = Shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    max_global_(input, output);
    return output;
}

void max_global_(Tensor input, Tensor output) {
    MaxGlobal::execute(input, output);
}

} // namespace infinicore::op
