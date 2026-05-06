#include "infinicore/ops/blas_amax.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<BlasAmax::schema> &BlasAmax::dispatcher() {
    static common::OpDispatcher<BlasAmax::schema> dispatcher_;
    return dispatcher_;
};

void BlasAmax::execute(Tensor result, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(result, x);
    infinicore::context::setDevice(result->device());
    dispatcher().lookup(result->device().getType())(result, x);
}

Tensor blas_amax(Tensor x) {
    auto result = Tensor::empty({}, DataType::I32, x->device());
    blas_amax_(result, x);
    return result;
}

void blas_amax_(Tensor result, Tensor x) {
    BlasAmax::execute(result, x);
}

} // namespace infinicore::op
