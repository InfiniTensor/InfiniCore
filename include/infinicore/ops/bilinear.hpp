#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Bilinear {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor out, Tensor x1, Tensor x2, Tensor weight, Tensor bias);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor bilinear(Tensor x1, Tensor x2, Tensor weight, Tensor bias);
void bilinear_(Tensor out, Tensor x1, Tensor x2, Tensor weight, Tensor bias);
} // namespace infinicore::op