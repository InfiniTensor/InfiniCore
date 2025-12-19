#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Mish {
public:
    using schema = void (*)(Tensor, Tensor, bool);
    static void execute(Tensor output, Tensor input, bool inplace);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor mish(Tensor input, bool inplace);
} // namespace infinicore::op
