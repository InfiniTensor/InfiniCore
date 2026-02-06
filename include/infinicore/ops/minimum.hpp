#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Minimum {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor input, Tensor other, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor minimum(Tensor input, Tensor other);
void minimum_(Tensor input, Tensor other, Tensor output);

} // namespace infinicore::op