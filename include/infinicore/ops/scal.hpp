#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Scal {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor alpha, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

void scal_(Tensor x, Tensor alpha);

} // namespace infinicore::op
