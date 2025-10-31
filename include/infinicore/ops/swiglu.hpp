#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class SwiGLU {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor swiglu(Tensor a, Tensor b);
void swiglu_(Tensor c, Tensor a, Tensor b);
} // namespace infinicore::op
