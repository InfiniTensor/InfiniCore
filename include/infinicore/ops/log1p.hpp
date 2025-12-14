#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Log1p {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor log1p(Tensor input);
void log1p_(Tensor output, Tensor input);
} // namespace infinicore::op
