#include "infinicore/ops/blas_copy.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<BlasCopy::schema> &BlasCopy::dispatcher() {
    static common::OpDispatcher<BlasCopy::schema> dispatcher_;
    return dispatcher_;
};

void BlasCopy::execute(Tensor x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(x, y);
}

void blas_copy_(Tensor x, Tensor y) {
    BlasCopy::execute(x, y);
}

} // namespace infinicore::op
