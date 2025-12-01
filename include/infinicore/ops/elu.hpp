#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Elu {
public:
    using schema = void (*)(Tensor, Tensor, float);
    static void execute(Tensor output, Tensor input, float alpha);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor elu(Tensor input, float alpha = 1.0f);
void elu_(Tensor output, Tensor input, float alpha = 1.0f);

} // namespace infinicore::op