#include "infinicore/ops/logical_xor.hpp"
#include "../../utils.hpp"
#include "infinicore/dtype.hpp"

namespace infinicore::op {

common::OpDispatcher<LogicalXor::schema> &LogicalXor::dispatcher() {
    static common::OpDispatcher<LogicalXor::schema> dispatcher_;
    return dispatcher_;
}

void LogicalXor::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor logical_xor(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), DataType::BOOL, a->device());
    logical_xor_(c, a, b);
    return c;
}

void logical_xor_(Tensor c, Tensor a, Tensor b) {
    LogicalXor::execute(c, a, b);
}
} // namespace infinicore::op
