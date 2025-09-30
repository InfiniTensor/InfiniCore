#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include "infinicore/common/utils.hpp"

namespace infinicore::op {
class Rearrange {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor rearrange(Tensor x);
void rearrange_(Tensor y, Tensor x);
} // namespace infinicore::op
