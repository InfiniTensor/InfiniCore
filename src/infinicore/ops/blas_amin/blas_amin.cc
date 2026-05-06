#include "infinicore/ops/blas_amin.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<BlasAmin::schema> &BlasAmin::dispatcher() {
    static common::OpDispatcher<BlasAmin::schema> dispatcher_;
    return dispatcher_;
};

void BlasAmin::execute(Tensor result, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(result, x);
    infinicore::context::setDevice(result->device());
    dispatcher().lookup(result->device().getType())(result, x);
}

Tensor blas_amin(Tensor x) {
    auto result = Tensor::empty({}, DataType::I32, x->device());
    blas_amin_(result, x);
    return result;
}

void blas_amin_(Tensor result, Tensor x) {
    BlasAmin::execute(result, x);
}

} // namespace infinicore::op
