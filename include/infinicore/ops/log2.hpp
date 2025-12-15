#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Log2 {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor log2(Tensor input);
void log2_(Tensor output, Tensor input);
} // namespace infinicore::op