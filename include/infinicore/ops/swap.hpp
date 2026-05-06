#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Swap {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor x, Tensor y);
    static common::OpDispatcher<schema> &dispatcher();
};

void swap_(Tensor x, Tensor y);

} // namespace infinicore::op
