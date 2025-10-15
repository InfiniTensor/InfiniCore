#include "infinicore/ops/rms_norm.hpp"

namespace infinicore::op {

common::OpDispatcher<RMSNorm::schema> &RMSNorm::dispatcher() {
    static common::OpDispatcher<RMSNorm::schema> dispatcher_;
    return dispatcher_;
};

void RMSNorm::execute(Tensor y, Tensor x, Tensor weight, float epsilon) {
    dispatcher().lookup(context::getDevice().getType())(y, x, weight, epsilon);
}

Tensor rms_norm(Tensor x, Tensor weight, float epsilon) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    rms_norm_(y, x, weight, epsilon);
    return y;
}

void rms_norm_(Tensor y, Tensor x, Tensor weight, float epsilon) {
    RMSNorm::execute(y, x, weight, epsilon);
}

} // namespace infinicore::op
