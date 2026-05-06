#include "infinicore/ops/rot.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rot::schema> &Rot::dispatcher() {
    static common::OpDispatcher<Rot::schema> dispatcher_;
    return dispatcher_;
};

void Rot::execute(Tensor x, Tensor y, Tensor c, Tensor s) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y, c, s);
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(x, y, c, s);
}

void rot_(Tensor x, Tensor y, Tensor c, Tensor s) {
    Rot::execute(x, y, c, s);
}

} // namespace infinicore::op
