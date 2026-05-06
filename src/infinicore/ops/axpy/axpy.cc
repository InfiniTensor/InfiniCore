#include "infinicore/ops/axpy.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Axpy::schema> &Axpy::dispatcher() {
    static common::OpDispatcher<Axpy::schema> dispatcher_;
    return dispatcher_;
};

void Axpy::execute(Tensor alpha, Tensor x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, y);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(alpha, x, y);
}

void axpy_(Tensor alpha, Tensor x, Tensor y) {
    Axpy::execute(alpha, x, y);
}

} // namespace infinicore::op
