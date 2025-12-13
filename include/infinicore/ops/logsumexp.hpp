#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class LogSumExp {
public:
    using schema = void (*)(Tensor, int, bool, Tensor);
    static void execute(Tensor input, int dim, bool keepdim, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor logsumexp(Tensor input, int dim, bool keepdim);
void logsumexp_(Tensor input, int dim, bool keepdim, Tensor output);
} // namespace infinicore::op
