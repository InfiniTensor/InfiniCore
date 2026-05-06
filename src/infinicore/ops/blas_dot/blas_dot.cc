#include "infinicore/ops/blas_dot.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<BlasDot::schema> &BlasDot::dispatcher() {
    static common::OpDispatcher<BlasDot::schema> dispatcher_;
    return dispatcher_;
};

void BlasDot::execute(Tensor result, Tensor x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(result, x, y);
    infinicore::context::setDevice(result->device());
    dispatcher().lookup(result->device().getType())(result, x, y);
}

Tensor blas_dot(Tensor x, Tensor y) {
    auto result = Tensor::empty({}, x->dtype(), x->device());
    blas_dot_(result, x, y);
    return result;
}

void blas_dot_(Tensor result, Tensor x, Tensor y) {
    BlasDot::execute(result, x, y);
}

} // namespace infinicore::op
