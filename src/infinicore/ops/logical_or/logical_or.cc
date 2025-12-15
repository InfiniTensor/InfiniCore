#include "infinicore/ops/logical_or.hpp"
#include "../../utils.hpp"
#include "infinicore/dtype.hpp"

namespace infinicore::op {

common::OpDispatcher<LogicalOr::schema> &LogicalOr::dispatcher() {
    static common::OpDispatcher<LogicalOr::schema> dispatcher_;
    return dispatcher_;
}

void LogicalOr::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor logical_or(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), DataType::BOOL, a->device());
    logical_or_(c, a, b);
    return c;
}

void logical_or_(Tensor c, Tensor a, Tensor b) {
    LogicalOr::execute(c, a, b);
}
} // namespace infinicore::op

