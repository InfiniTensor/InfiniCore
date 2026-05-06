#include "infinicore/ops/asum.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Asum::schema> &Asum::dispatcher() {
    static common::OpDispatcher<Asum::schema> dispatcher_;
    return dispatcher_;
};

void Asum::execute(Tensor result, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(result, x);
    infinicore::context::setDevice(result->device());
    dispatcher().lookup(result->device().getType())(result, x);
}

Tensor asum(Tensor x) {
    auto result = Tensor::empty({}, x->dtype(), x->device());
    asum_(result, x);
    return result;
}

void asum_(Tensor result, Tensor x) {
    Asum::execute(result, x);
}

} // namespace infinicore::op
