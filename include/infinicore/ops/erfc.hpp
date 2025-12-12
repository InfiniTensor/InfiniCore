#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Erfc {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor erfc(Tensor input);
void erfc_(Tensor output, Tensor input);
} // namespace infinicore::op

