#include "infinicore/ops/mish.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Mish::schema> &Mish::dispatcher() {
    static common::OpDispatcher<Mish::schema> dispatcher_;
    return dispatcher_;
};

void Mish::execute(Tensor output, Tensor input, bool inplace) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    dispatcher().lookup(output->device().getType())(output, input, inplace);
}

Tensor mish(Tensor input, bool inplace) {
    if(inplace) {
        Mish::execute(input, input, inplace);
        return input;
    }

    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    Mish::execute(output, input, inplace);
    return output;
}

} // namespace infinicore::op
