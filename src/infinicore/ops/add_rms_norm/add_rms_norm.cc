#include "infinicore/ops/add_rms_norm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<AddRMSNorm::schema> &AddRMSNorm::dispatcher() {
    static common::OpDispatcher<AddRMSNorm::schema> dispatcher_;
    return dispatcher_;
};

void AddRMSNorm::execute(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, residual_out, a, b, weight);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, residual_out, a, b, weight, epsilon);
}

std::pair<Tensor, Tensor> add_rms_norm(Tensor a, Tensor b, Tensor weight, float epsilon) {
    auto y = Tensor::empty(a->shape(), a->dtype(), a->device());
    auto residual_out = Tensor::empty(a->shape(), a->dtype(), a->device());
    add_rms_norm_(y, residual_out, a, b, weight, epsilon);
    return std::make_pair(y, residual_out);
}

void add_rms_norm_(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon) {
    AddRMSNorm::execute(y, residual_out, a, b, weight, epsilon);
}

} // namespace infinicore::op
