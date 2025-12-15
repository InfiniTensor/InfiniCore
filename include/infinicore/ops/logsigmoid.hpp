#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class LogSigmoid {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor logsigmoid(Tensor input);
void logsigmoid_(Tensor output, Tensor input);
} // namespace infinicore::op

