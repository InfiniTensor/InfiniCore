#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Round {
public:
    using schema = void (*)(Tensor, Tensor, int);
    static void execute(Tensor y, Tensor x, int decimals);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor round(Tensor x, int decimals = 0);
void round_(Tensor y, Tensor x, int decimals = 0);
} // namespace infinicore::op