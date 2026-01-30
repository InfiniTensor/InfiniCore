#include "infinicore/ops/scaled_mm_i8.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<ScaledMMI8::schema> &ScaledMMI8::dispatcher() {
    static common::OpDispatcher<ScaledMMI8::schema> dispatcher_;
    return dispatcher_;
};

void ScaledMMI8::execute(Tensor c, Tensor a_p, Tensor a_s, Tensor b_p, Tensor b_s, std::optional<Tensor> bias) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a_p, a_s, b_p, b_s);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a_p, a_s, b_p, b_s, bias);
}

void scaled_mm_i8_(Tensor c, Tensor a_p, Tensor a_s, Tensor b_p, Tensor b_s, std::optional<Tensor> bias) {
    ScaledMMI8::execute(c, a_p, a_s, b_p, b_s, bias);
}

} // namespace infinicore::op
