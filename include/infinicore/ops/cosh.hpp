#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Cosh {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor cosh(Tensor x);
void cosh_(Tensor y, Tensor x);
} // namespace infinicore::op