#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Maximum {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor maximum(Tensor a, Tensor b);
void maximum_(Tensor c, Tensor a, Tensor b);
} // namespace infinicore::op
