#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Log10 {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor log10(Tensor input);
void log10_(Tensor output, Tensor input);
} // namespace infinicore::op
