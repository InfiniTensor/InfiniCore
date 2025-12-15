#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class LogicalOr {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor logical_or(Tensor a, Tensor b);
void logical_or_(Tensor c, Tensor a, Tensor b);
} // namespace infinicore::op

