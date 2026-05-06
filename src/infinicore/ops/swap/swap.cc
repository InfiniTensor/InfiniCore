#include "infinicore/ops/swap.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Swap::schema> &Swap::dispatcher() {
    static common::OpDispatcher<Swap::schema> dispatcher_;
    return dispatcher_;
};

void Swap::execute(Tensor x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y);
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(x, y);
}

void swap_(Tensor x, Tensor y) {
    Swap::execute(x, y);
}

} // namespace infinicore::op
