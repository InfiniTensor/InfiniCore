#include "infinicore/ops/per_channel_quant_i8.hpp"

#include "../../utils.hpp"
#include <iostream>

namespace infinicore::op {

common::OpDispatcher<PerChannelQuantI8::schema> &PerChannelQuantI8::dispatcher() {
    static common::OpDispatcher<PerChannelQuantI8::schema> dispatcher_;
    return dispatcher_;
};

void PerChannelQuantI8::execute(Tensor x, Tensor x_packed, Tensor x_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, x_packed, x_scale);
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(x, x_packed, x_scale);
}

void per_channel_quant_i8_(Tensor x, Tensor x_packed, Tensor x_scale) {
    PerChannelQuantI8::execute(x, x_packed, x_scale);
}
} // namespace infinicore::op
