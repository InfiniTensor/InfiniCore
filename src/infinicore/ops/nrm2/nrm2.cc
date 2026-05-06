#include "infinicore/ops/nrm2.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Nrm2::schema> &Nrm2::dispatcher() {
    static common::OpDispatcher<Nrm2::schema> dispatcher_;
    return dispatcher_;
};

void Nrm2::execute(Tensor result, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(result, x);
    infinicore::context::setDevice(result->device());
    dispatcher().lookup(result->device().getType())(result, x);
}

Tensor nrm2(Tensor x) {
    auto result = Tensor::empty({}, x->dtype(), x->device());
    nrm2_(result, x);
    return result;
}

void nrm2_(Tensor result, Tensor x) {
    Nrm2::execute(result, x);
}

} // namespace infinicore::op
