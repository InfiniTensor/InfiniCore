#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Sigmoid {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor sigmoid(Tensor input);
void sigmoid_(Tensor output, Tensor input);
} // namespace infinicore::op

