#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Add {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor add(Tensor a, Tensor b);
void add_(Tensor c, Tensor a, Tensor b);
Tensor operator+(Tensor a, Tensor b);
} // namespace infinicore::op
