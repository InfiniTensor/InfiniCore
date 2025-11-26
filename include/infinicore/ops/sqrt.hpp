#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Sqrt {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor sqrt(Tensor input);
void sqrt_(Tensor ouput, Tensor input);

} // namespace infinicore::op