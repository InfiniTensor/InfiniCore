#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class RMSNorm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, float);
    static void execute(Tensor y, Tensor x, Tensor weight, float epsilon = 1e-5f);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor rms_norm(Tensor x, Tensor weight, float epsilon = 1e-5f);
void rms_norm_(Tensor y, Tensor x, Tensor weight, float epsilon = 1e-5f);
} // namespace infinicore::op
