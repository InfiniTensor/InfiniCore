#include "infinicore/ops/dequantize_awq.hpp"

#include "../../utils.hpp"
#include <iostream>

namespace infinicore::op {

common::OpDispatcher<DequantizeAWQ::schema> &DequantizeAWQ::dispatcher() {
    static common::OpDispatcher<DequantizeAWQ::schema> dispatcher_;
    return dispatcher_;
};

void DequantizeAWQ::execute(Tensor x, Tensor x_packed, Tensor x_scale, Tensor x_zeros) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, x_packed, x_scale, x_zeros);
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().getType())(x, x_packed, x_scale, x_zeros);
}

void dequantize_awq_(Tensor x, Tensor x_packed, Tensor x_scale, Tensor x_zeros) {
    DequantizeAWQ::execute(x, x_packed, x_scale, x_zeros);
}
} // namespace infinicore::op