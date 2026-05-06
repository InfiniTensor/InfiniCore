#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Asum {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor result, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor asum(Tensor x);
void asum_(Tensor result, Tensor x);

} // namespace infinicore::op
