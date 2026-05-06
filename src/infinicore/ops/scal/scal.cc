#include "infinicore/ops/scal.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Scal::schema> &Scal::dispatcher() {
    static common::OpDispatcher<Scal::schema> dispatcher_;
    return dispatcher_;
};

void Scal::execute(Tensor alpha, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x);
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(alpha, x);
}

void scal_(Tensor x, Tensor alpha) {
    Scal::execute(alpha, x);
}

} // namespace infinicore::op
