#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Rot {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor x, Tensor y, Tensor c, Tensor s);
    static common::OpDispatcher<schema> &dispatcher();
};

void rot_(Tensor x, Tensor y, Tensor c, Tensor s);

} // namespace infinicore::op
